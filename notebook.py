import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from jax import nn as jnn, lax, vmap
    import flax.linen as nn
    import optax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive
    from numpyro.contrib.module import flax_module
    from PIL import Image
    import matplotlib.pyplot as plt

    jax.config.update("jax_enable_x64", False)
    print("devices:", jax.devices())

    return (
        Image,
        Predictive,
        dist,
        flax_module,
        jax,
        jnn,
        jnp,
        jr,
        lax,
        nn,
        np,
        numpyro,
        optax,
        plt,
        vmap,
    )


@app.cell(hide_code=True)
def _(Image, jnp, jr, lax, np, vmap):
    # --- Data loading & tiling -------------------------------------------------
    # Load the single real image, normalize to [0,1], and produce a deterministic
    # 3x3 grid of crops at native resolution (no downsampling). Smaller crops also
    # let us augment by random origin jitter -- effectively unlimited training tiles.

    _img_pil = Image.open("example.jpg").convert("RGB")
    img_full = jnp.asarray(np.array(_img_pil), dtype=jnp.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    print("full image:", img_full.shape)

    # Tile size kept small to stay within 8 GB VRAM with N_SLOTS droplets.
    # Pixels are native-resolution; we just take smaller windows.
    TILE_H, TILE_W = 96, 96
    GRID = 3

    def deterministic_tiles(img, grid=GRID, tile_h=TILE_H, tile_w=TILE_W):
        H, W = img.shape[:2]
        ys = np.linspace(0, H - tile_h, grid).astype(int)
        xs = np.linspace(0, W - tile_w, grid).astype(int)
        out = []
        for y in ys:
            for x in xs:
                out.append(img[y:y+tile_h, x:x+tile_w])
        return jnp.stack(out, axis=0)

    def random_crop(img, key, tile_h=TILE_H, tile_w=TILE_W):
        H, W = img.shape[:2]
        ky, kx = jr.split(key)
        y = jr.randint(ky, (), 0, H - tile_h + 1)
        x = jr.randint(kx, (), 0, W - tile_w + 1)
        return lax.dynamic_slice(img, (y, x, 0), (tile_h, tile_w, 3))

    def random_crop_batch(img, key, batch, tile_h=TILE_H, tile_w=TILE_W):
        keys = jr.split(key, batch)
        return vmap(lambda k: random_crop(img, k, tile_h, tile_w))(keys)

    real_tiles = deterministic_tiles(img_full)
    print("real tiles:", real_tiles.shape)

    return TILE_H, TILE_W, real_tiles


@app.cell(hide_code=True)
def _(np, plt, real_tiles):
    # Sanity check: show the 9 real tiles
    def _show_real_tiles():
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for ax, tile in zip(axes.ravel(), np.asarray(real_tiles)):
            ax.imshow(tile)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Real tiles (3x3 deterministic grid)")
        fig.tight_layout()
        return fig
    _show_real_tiles()

    return


@app.cell(hide_code=True)
def _(TILE_H, TILE_W, jnp):
    # --- Global config ---------------------------------------------------------
    N_SLOTS = 32
    K_COMP = 3
    EMB_DIM = 128
    LIK_HIDDEN = 64
    POST_HIDDEN = 128

    # Derived dimensions (used by flax modules and unconstrained transforms)
    PER_SLOT_DIMS = 2 + 1 + 1 + 1 + K_COMP + 1   # cy,cx,ls,la,th,comp(K),pres
    BG_DIMS = 4 * 3
    LIK_IN = 3 + K_COMP + 2 + 1                  # bg(3)+comp(K)+offset(2)+log_scale(1)

    LOG_SCALE_MIN, LOG_SCALE_MAX = jnp.log(2.0), jnp.log(30.0)

    print(f"slots={N_SLOTS}, K={K_COMP}, tile={TILE_H}x{TILE_W}")

    return (
        BG_DIMS,
        EMB_DIM,
        K_COMP,
        LIK_HIDDEN,
        LOG_SCALE_MAX,
        LOG_SCALE_MIN,
        N_SLOTS,
        PER_SLOT_DIMS,
        POST_HIDDEN,
    )


@app.cell(hide_code=True)
def _(
    BG_DIMS,
    EMB_DIM,
    LIK_HIDDEN,
    LOG_SCALE_MAX,
    LOG_SCALE_MIN,
    N_SLOTS,
    PER_SLOT_DIMS,
    POST_HIDDEN,
    TILE_H,
    TILE_W,
    jnp,
    nn,
):
    # --- Flax modules ---------------------------------------------------------
    # Three small nets:
    #   Encoder        : (H, W, 3) -> (EMB_DIM,)         3-block stride-2 CNN + linear
    #   PosteriorHead  : (EMB_DIM,) -> (bg head, slot head)
    #   LikelihoodMLP  : per-pixel droplet residual, FACTORED as
    #                       delta = colour_head(comp) * spatial_gain(bg, offset, log_scale)
    #                    so that the "what" (composition -> colour) and the "where"
    #                    (spatial profile) cannot leak into each other.
    #                    This dodges the prior-predictive failure mode where a
    #                    nuisance-feature bias dominates the comp signal.

    class Encoder(nn.Module):
        emb_dim: int = EMB_DIM
        @nn.compact
        def __call__(self, x):
            x = x[None]
            for c in (16, 32, 64):
                x = nn.Conv(c, (3, 3), strides=(2, 2), padding="SAME")(x)
                x = nn.gelu(x)
            x = x.mean(axis=(1, 2))[0]
            x = nn.Dense(self.emb_dim)(x)
            return x

    class PosteriorHead(nn.Module):
        n_slots: int
        per_slot_dims: int
        bg_dims: int
        hidden: int = POST_HIDDEN
        @nn.compact
        def __call__(self, emb):
            b = nn.Dense(self.hidden)(emb); b = nn.gelu(b)
            bg = nn.Dense(2 * self.bg_dims)(b).reshape(self.bg_dims, 2)
            s = nn.Dense(self.hidden)(emb); s = nn.gelu(s)
            slots = nn.Dense(self.n_slots * 2 * self.per_slot_dims)(s)
            slots = slots.reshape(self.n_slots, self.per_slot_dims, 2)
            return bg, slots

    class LikelihoodMLP(nn.Module):
        """Factored droplet appearance:
           delta = colour(comp) * spatial(bg, offset_norm, log_scale)
        where colour: R^K -> R^3 and spatial: R^(3+2+1) -> R (scalar gain).
        Inputs are normalised to roughly O(1) before the MLPs so init balances.

        Args to __call__:
           bg          : (..., 3)
           comp        : (..., K)
           offset_norm : (..., 2)        already in droplet-local frame
           log_scale   : (...,)          raw log_scale value
        """
        hidden: int = LIK_HIDDEN
        @nn.compact
        def __call__(self, bg, comp, offset_norm, log_scale):
            # Normalise nuisance features
            bg_n = (bg - 0.5) * 2.0                        # bg in [0,1] -> ~[-1,1]
            # offset_norm is already O(1) (Mahalanobis scaled), keep
            ls_mid = 0.5 * (LOG_SCALE_MIN + LOG_SCALE_MAX)
            ls_range = 0.5 * (LOG_SCALE_MAX - LOG_SCALE_MIN)
            ls_n = (log_scale - ls_mid) / ls_range          # ~[-1,1]
            ls_n = ls_n[..., None] if jnp.ndim(ls_n) == jnp.ndim(bg_n) - 1 else ls_n

            # Colour head: comp -> R^3 (bias-free final so comp=0 -> colour=0)
            c = nn.Dense(self.hidden)(comp); c = nn.gelu(c)
            c = nn.Dense(self.hidden)(c); c = nn.gelu(c)
            colour = nn.Dense(3, use_bias=False)(c)         # (..., 3)

            # Spatial head: (bg_n, offset_norm, ls_n) -> R (scalar gain)
            spatial_in = jnp.concatenate([bg_n, offset_norm, ls_n], axis=-1)
            s = nn.Dense(self.hidden)(spatial_in); s = nn.gelu(s)
            s = nn.Dense(self.hidden)(s); s = nn.gelu(s)
            # final scaled init + softplus bias so initial gain is ~1 (visible droplet)
            gain = nn.Dense(1, kernel_init=nn.initializers.variance_scaling(0.3**2, "fan_in", "uniform"),
                            bias_init=nn.initializers.constant(0.5))(s)        # (..., 1)
            return colour * gain                            # (..., 3)

    encoder_module = Encoder()
    posterior_module = PosteriorHead(
        n_slots=N_SLOTS, per_slot_dims=PER_SLOT_DIMS, bg_dims=BG_DIMS,
    )
    likelihood_module = LikelihoodMLP()

    # Flax input shape probes for module init
    ENC_IN_SHAPE = (TILE_H, TILE_W, 3)
    POST_IN_SHAPE = (EMB_DIM,)
    # Likelihood now has 4 separate input args -> we pass init args directly
    # below via flax_module(*args).
    print("flax modules defined (factored likelihood: colour * spatial)")

    return (
        ENC_IN_SHAPE,
        POST_IN_SHAPE,
        encoder_module,
        likelihood_module,
        posterior_module,
    )


@app.cell(hide_code=True)
def _(jnp):
    # --- Per-droplet pixel update ---------------------------------------------
    # Same invariants as before:
    #   out = bg + presence * env(||offset_norm||) * delta(bg, comp, offset, log_scale)
    # delta now comes from the factored LikelihoodMLP (colour*spatial gain).

    def apply_droplet(lik_apply, bg, comp, offset_norm, log_scale, presence):
        """lik_apply: callable (bg, comp, offset_norm, log_scale) -> delta R^3."""
        delta = lik_apply(bg, comp, offset_norm, log_scale)
        r2 = (offset_norm ** 2).sum(axis=-1)
        env = jnp.exp(-0.5 * r2)
        gate = (presence * env)[..., None]
        return bg + gate * delta


    return


@app.cell(hide_code=True)
def _(
    K_COMP,
    LOG_SCALE_MAX,
    LOG_SCALE_MIN,
    N_SLOTS,
    TILE_H,
    TILE_W,
    dist,
    jax,
    jnp,
    lax,
    numpyro,
):
    # --- Background prior + simulator ----------------------------------------
    def render_background(bg_corners, H, W):
        ys = jnp.linspace(0.0, 1.0, H); xs = jnp.linspace(0.0, 1.0, W)
        Y, X = jnp.meshgrid(ys, xs, indexing="ij")
        tl, tr, bl, br = bg_corners[0], bg_corners[1], bg_corners[2], bg_corners[3]
        top = (1 - X)[..., None] * tl + X[..., None] * tr
        bot = (1 - X)[..., None] * bl + X[..., None] * br
        return (1 - Y)[..., None] * top + Y[..., None] * bot

    def droplet_L(log_scale, log_aspect, theta):
        s = jnp.exp(log_scale); a = jnp.exp(log_aspect)
        sx, sy = s * a, s / a
        c, sn = jnp.cos(theta), jnp.sin(theta)
        R = jnp.array([[c, -sn], [sn, c]])
        D = jnp.diag(jnp.array([1.0 / sx, 1.0 / sy]))
        return D @ R.T

    OVERLAP_R = 6.0
    OVERLAP_W = 4.0

    def overlap_penalty(centers, presence, r=OVERLAP_R):
        """Pairwise Gaussian repulsion, presence-gated. O(N^2)."""
        diff = centers[:, None, :] - centers[None, :, :]
        d2 = (diff ** 2).sum(-1)
        pair = jnp.exp(-d2 / (2.0 * r ** 2))
        mask = presence[:, None] * presence[None, :]
        triu = jnp.triu(jnp.ones_like(pair), k=1)
        return (pair * mask * triu).sum()

    def simulate_tile(lik_apply, latents, H=TILE_H, W=TILE_W):
        """lik_apply: callable feats -> delta (the flax-bound likelihood net).
        Returns unclipped (H, W, 3); clip outside for display."""
        bg = render_background(latents["bg_corners"], H, W)
        ys = jnp.arange(H, dtype=jnp.float32); xs = jnp.arange(W, dtype=jnp.float32)
        Y, X = jnp.meshgrid(ys, xs, indexing="ij")
        px = jnp.stack([Y, X], axis=-1)

        def add_droplet(img, slot):
            center, ls, la, th, comp, pres = slot
            L = droplet_L(ls, la, th)
            offset_norm = (px - center) @ L.T
            # broadcast per-droplet scalars to per-pixel
            comp_b = jnp.broadcast_to(comp, (H, W, K_COMP))
            ls_b = jnp.broadcast_to(ls, (H, W))
            delta = lik_apply(img, comp_b, offset_norm, ls_b)
            env = jnp.exp(-0.5 * (offset_norm ** 2).sum(-1))
            new = img + (pres * env)[..., None] * delta
            return new, None

        slots = (latents["centers"], latents["log_scale"], latents["log_aspect"],
                 latents["theta"], latents["comp"], latents["presence"])
        img, _ = lax.scan(jax.checkpoint(add_droplet), bg, slots)
        return img

    # --- numpyro prior model ---------------------------------------------------
    # All latent priors live here; Predictive(prior_model) gives ancestral samples.
    # The overlap factor makes the joint prior repulsive (and is also added to the
    # training loss on posterior-mean latents -- see KDsH).

    def prior_model(H=TILE_H, W=TILE_W, n_slots=N_SLOTS):
        bg_corners = numpyro.sample(
            "bg_corners",
            dist.Normal(jnp.full((4, 3), 0.6), 0.15),
        )
        with numpyro.plate("slots", n_slots):
            cy = numpyro.sample("cy", dist.Uniform(0.0, float(H)))
            cx = numpyro.sample("cx", dist.Uniform(0.0, float(W)))
            log_scale = numpyro.sample("log_scale",
                                       dist.Uniform(LOG_SCALE_MIN, LOG_SCALE_MAX))
            log_aspect = numpyro.sample("log_aspect", dist.Normal(0.0, 0.25))
            theta = numpyro.sample("theta", dist.Uniform(-jnp.pi/2, jnp.pi/2))
            comp = numpyro.sample("comp",
                                  dist.Normal(jnp.zeros(K_COMP), 1.0).to_event(1))
            presence = numpyro.sample("presence", dist.Beta(0.3, 0.3))
        centers = jnp.stack([cy, cx], axis=-1)
        numpyro.factor("overlap", -OVERLAP_W * overlap_penalty(centers, presence))
        return dict(
            bg_corners=bg_corners, centers=centers,
            log_scale=log_scale, log_aspect=log_aspect, theta=theta,
            comp=comp, presence=presence,
        )

    def pack_latents(samples):
        return dict(
            bg_corners=samples["bg_corners"],
            centers=jnp.stack([samples["cy"], samples["cx"]], axis=-1),
            log_scale=samples["log_scale"],
            log_aspect=samples["log_aspect"],
            theta=samples["theta"],
            comp=samples["comp"],
            presence=samples["presence"],
        )

    OBS_SIGMA = 0.03
    print("simulator + prior defined (flax-aware)")

    return OBS_SIGMA, overlap_penalty, pack_latents, prior_model, simulate_tile


@app.cell(hide_code=True)
def _(
    K_COMP,
    Predictive,
    jax,
    jnp,
    jr,
    likelihood_module,
    np,
    pack_latents,
    plt,
    prior_model,
    simulate_tile,
):
    # --- Prior predictive check ----------------------------------------------
    import time

    def _init_lik_params(key):
        """Initialise just the likelihood module params for prior-predictive runs
        BEFORE training begins."""
        return likelihood_module.init(
            key, jnp.zeros((3,)), jnp.zeros((K_COMP,)), jnp.zeros((2,)), jnp.zeros(()),
        )

    lik_params_init = _init_lik_params(jr.PRNGKey(42))

    @jax.jit
    def _sample_and_sim(key, lik_params):
        pred = Predictive(prior_model, num_samples=1)
        samples = pred(key)
        flat = {k: v[0] for k, v in samples.items()}
        packed = pack_latents(flat)
        lik = lambda bg, comp, off, ls: likelihood_module.apply(lik_params, bg, comp, off, ls)
        img = simulate_tile(lik, packed)
        return img, packed

    def _do_ppc():
        t0 = time.time()
        imgs = []
        keys = jr.split(jr.PRNGKey(7), 12)
        for k in keys:
            img, _ = _sample_and_sim(k, lik_params_init)
            imgs.append(np.asarray(jnp.clip(img, 0.0, 1.0)))
        print(f"simulated 12 tiles in {time.time()-t0:.2f}s")
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax, im in zip(axes.ravel(), imgs):
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Prior predictive (random likelihood net)")
        fig.tight_layout()
        return fig
    _do_ppc()

    return


@app.cell(hide_code=True)
def _(
    BG_DIMS,
    K_COMP,
    LOG_SCALE_MAX,
    LOG_SCALE_MIN,
    TILE_H,
    TILE_W,
    jax,
    jnn,
    jnp,
):
    # --- Unconstrained <-> constrained latent transforms ----------------------

    def _logit(x, eps=1e-6):
        x = jnp.clip(x, eps, 1.0 - eps)
        return jnp.log(x) - jnp.log1p(-x)

    def true_latents_to_unconstrained(latents):
        bg = latents["bg_corners"].reshape(BG_DIMS)
        cy = _logit(latents["centers"][:, 0] / TILE_H)
        cx = _logit(latents["centers"][:, 1] / TILE_W)
        ls = _logit((latents["log_scale"] - LOG_SCALE_MIN) / (LOG_SCALE_MAX - LOG_SCALE_MIN))
        la = latents["log_aspect"]
        th = _logit((latents["theta"] + jnp.pi/2) / jnp.pi)
        pres = _logit(latents["presence"])
        per_slot = jnp.concatenate([
            cy[:, None], cx[:, None], ls[:, None], la[:, None],
            th[:, None], latents["comp"], pres[:, None],
        ], axis=-1)
        return bg, per_slot

    def unconstrained_to_latents(bg_u, slots_u):
        bg_corners = bg_u.reshape(4, 3)
        cy = jnn.sigmoid(slots_u[..., 0]) * TILE_H
        cx = jnn.sigmoid(slots_u[..., 1]) * TILE_W
        ls = LOG_SCALE_MIN + jnn.sigmoid(slots_u[..., 2]) * (LOG_SCALE_MAX - LOG_SCALE_MIN)
        la = slots_u[..., 3]
        th = jnn.sigmoid(slots_u[..., 4]) * jnp.pi - jnp.pi/2
        comp = slots_u[..., 5:5+K_COMP]
        pres = jnn.sigmoid(slots_u[..., 5+K_COMP])
        return dict(
            bg_corners=bg_corners,
            centers=jnp.stack([cy, cx], axis=-1),
            log_scale=ls, log_aspect=la, theta=th, comp=comp, presence=pres,
        )

    # --- Permutation-invariant NPE log-prob (DETR-style soft Hungarian) -------
    LOG_SIGMA_CLAMP = (-4.0, 2.0)

    def post_log_prob_set(bg_head, slot_head, bg_u, slots_u, present_mask):
        mu_b = bg_head[:, 0]; ls_b = jnp.clip(bg_head[:, 1], *LOG_SIGMA_CLAMP)
        mu_s = slot_head[..., 0]; ls_s = jnp.clip(slot_head[..., 1], *LOG_SIGMA_CLAMP)
        sigma_b = jnp.exp(ls_b)
        lp_bg = (-0.5 * ((bg_u - mu_b) / sigma_b) ** 2 - ls_b - 0.5*jnp.log(2*jnp.pi)).sum()
        diff = slots_u[:, None, :] - mu_s[None, :, :]
        z2 = (diff / jnp.exp(ls_s)[None, :, :]) ** 2
        lp_pair = (-0.5 * z2 - ls_s[None, :, :] - 0.5*jnp.log(2*jnp.pi)).sum(-1)  # (Nt, Np)
        Np = mu_s.shape[0]
        lp_per_true = jax.scipy.special.logsumexp(lp_pair, axis=1) - jnp.log(Np)
        return lp_bg + (lp_per_true * present_mask).sum()

    print("unconstrained transforms + set-NPE log-prob ready")

    return (
        post_log_prob_set,
        true_latents_to_unconstrained,
        unconstrained_to_latents,
    )


@app.cell(hide_code=True)
def _(
    ENC_IN_SHAPE,
    K_COMP,
    POST_IN_SHAPE,
    encoder_module,
    flax_module,
    jax,
    jnp,
    likelihood_module,
    numpyro,
    overlap_penalty,
    post_log_prob_set,
    posterior_module,
    simulate_tile,
    true_latents_to_unconstrained,
    unconstrained_to_latents,
    vmap,
):
    # --- Training: numpyro-native NPE + recon + overlap -----------------------
    #
    # We use numpyro's parameter store as the single source of truth for all flax
    # module parameters via flax_module. The "training model" is a numpyro model
    # that runs all three terms as numpyro.factor calls; we then minimise its
    # negative joint log-prob with optax (via numpyro.optim is not strictly needed,
    # but flax_module ensures params are discoverable).
    #
    # This is simpler than wiring SVI explicitly because our loss is not an ELBO;
    # we just want gradient descent on  -log_p_factors  w.r.t. the flax params.

    BATCH = 8
    LAMBDA_REC = 15000.0
    LAMBDA_OVERLAP = 50.0
    LR = 3e-4

    def _bind_modules():
        """Inside a numpyro model context, register the three flax modules and
        return their bound apply-functions."""
        encoder = flax_module("encoder", encoder_module, input_shape=ENC_IN_SHAPE)
        posterior = flax_module("posterior", posterior_module, input_shape=POST_IN_SHAPE)
        # Likelihood takes 4 named inputs; supply dummy positional init args.
        likelihood = flax_module(
            "likelihood", likelihood_module,
            jnp.zeros((3,)), jnp.zeros((K_COMP,)), jnp.zeros((2,)), jnp.zeros(()),
        )
        return encoder, posterior, likelihood

    def npe_loss_one(encoder, posterior, latents, image):
        emb = encoder(image)
        bg_head, slot_head = posterior(emb)
        bg_u, slots_u = true_latents_to_unconstrained(latents)
        present_mask = (latents["presence"] > 0.5).astype(jnp.float32)
        return -post_log_prob_set(bg_head, slot_head, bg_u, slots_u, present_mask)

    def recon_and_overlap_one(encoder, posterior, likelihood, image):
        emb = encoder(image)
        bg_head, slot_head = posterior(emb)
        bg_u = bg_head[:, 0]
        slots_u = slot_head[..., 0]
        latents = unconstrained_to_latents(bg_u, slots_u)
        sim = simulate_tile(likelihood, latents)
        mse = jnp.mean((sim - image) ** 2)
        ov = overlap_penalty(latents["centers"], latents["presence"])
        return mse, ov

    def training_model(sim_latents, sim_images, real_images,
                       lambda_rec=LAMBDA_REC, lambda_ov=LAMBDA_OVERLAP):
        """numpyro model whose log-prob equals -[NPE + lambda_rec*REC + lambda_ov*OV].
        Weights are passable so an outer training loop can run a curriculum
        (e.g. lambda_rec=0 for pure NPE warmup)."""
        encoder, posterior, likelihood = _bind_modules()

        def per_npe(i, img):
            latents_i = jax.tree_util.tree_map(lambda v: v[i], sim_latents)
            return npe_loss_one(encoder, posterior, latents_i, img)
        L_npe = vmap(per_npe)(jnp.arange(sim_images.shape[0]), sim_images).mean()

        def per_rec(img):
            return recon_and_overlap_one(encoder, posterior, likelihood, img)
        rec_mses, ovs = vmap(per_rec)(real_images)
        L_rec = rec_mses.mean()
        L_ov = ovs.mean()

        numpyro.factor("L_npe", -L_npe)
        numpyro.factor("L_rec", -lambda_rec * L_rec)
        numpyro.factor("L_overlap", -lambda_ov * L_ov)
        numpyro.deterministic("aux_L_npe", L_npe)
        numpyro.deterministic("aux_L_rec", L_rec)
        numpyro.deterministic("aux_L_overlap", L_ov)

    # --- SVI setup -------------------------------------------------------------
    # Use AutoDelta on the (empty) latent space; flax_module params are
    # auto-discovered. Loss is Trace_ELBO of the training_model -> equals the
    # total loss above. This is the standard numpyro pattern for amortized
    # inference with flax modules.

    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta

    def make_guide():
        return AutoDelta(training_model)

    # We will create the SVI state inside the loop cell, so that we can re-run
    # the loop and pick up where we left off via the SVI state object.
    print("training_model ready")

    return BATCH, LAMBDA_OVERLAP, LAMBDA_REC, LR, training_model


@app.cell(hide_code=True)
def _(
    BATCH,
    LAMBDA_OVERLAP,
    LAMBDA_REC,
    LR,
    OBS_SIGMA,
    Predictive,
    TILE_H,
    TILE_W,
    jax,
    jnp,
    jr,
    likelihood_module,
    numpyro,
    optax,
    pack_latents,
    prior_model,
    real_tiles,
    simulate_tile,
    training_model,
    vmap,
):
    # --- Training loop --------------------------------------------------------
    import time as _time

    N_STEPS = 5000
    LOG_EVERY = 200
    # Curriculum: pure NPE for the first WARMUP steps, then linearly ramp
    # in recon and overlap losses to their full values by RAMP_END.
    WARMUP_END = 500
    RAMP_END = 2500

    def sim_batch(key, likelihood, B):
        pred = Predictive(prior_model, num_samples=B)
        samples = pred(key)
        packed = pack_latents(samples)
        def one(i):
            single = jax.tree_util.tree_map(lambda v: v[i], packed)
            return simulate_tile(likelihood, single)
        imgs = vmap(one)(jnp.arange(B))
        nk = jr.fold_in(key, 1)
        return packed, jnp.clip(imgs + OBS_SIGMA * jr.normal(nk, imgs.shape), 0.0, 1.0)

    # --- Pure-JAX gradient step (no SVI overhead). Still uses flax_module for
    # parameter registration: we extract params from numpyro init once, then do
    # our own optax loop. Simpler and ~3x faster than SVI.update for this case.

    def _init_params(key):
        """Initialise all flax params by running the model once under numpyro
        trace; flax_module sites end up as numpyro.param sites, which we collect."""
        # Dummy data to trigger flax module initialisations
        dummy_lats = pack_latents(Predictive(prior_model, num_samples=BATCH)(key))
        dummy_imgs = jr.uniform(jr.fold_in(key, 2), (BATCH, TILE_H, TILE_W, 3))
        with numpyro.handlers.trace() as tr, numpyro.handlers.seed(rng_seed=0):
            training_model(dummy_lats, dummy_imgs, dummy_imgs)
        out = {}
        for name, site in tr.items():
            if site["type"] == "param":
                out[name] = site["value"]
        return out

    def loss_fn(params, sim_lats, sim_imgs, real_imgs, lam_rec, lam_ov):
        """Compute total loss + auxiliaries with curriculum weights."""
        handler = numpyro.handlers.substitute(
            numpyro.handlers.seed(training_model, rng_seed=0),
            data=params,
        )
        with numpyro.handlers.trace() as tr:
            handler(sim_lats, sim_imgs, real_imgs, lam_rec, lam_ov)
        L_npe = tr["aux_L_npe"]["value"]
        L_rec = tr["aux_L_rec"]["value"]
        L_ov  = tr["aux_L_overlap"]["value"]
        total = L_npe + lam_rec * L_rec + lam_ov * L_ov
        return total, (L_npe, L_rec, L_ov)

    _optimizer = optax.adam(LR)

    @jax.jit
    def _step(params, opt_state, key, real_imgs, lam_rec, lam_ov):
        lik_params_pytree = {"params": params["likelihood$params"]}
        lik = lambda bg, comp, off, ls: likelihood_module.apply(lik_params_pytree, bg, comp, off, ls)
        sim_lats, sim_imgs = sim_batch(key, lik, BATCH)
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, sim_lats, sim_imgs, real_imgs, lam_rec, lam_ov,
        )
        updates, opt_state = _optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, total, aux

    def schedule(step):
        """Curriculum: pure NPE for [0, WARMUP_END), linear ramp on [WARMUP_END, RAMP_END),
        full weights afterwards."""
        if step < WARMUP_END:
            f = 0.0
        elif step < RAMP_END:
            f = (step - WARMUP_END) / float(RAMP_END - WARMUP_END)
        else:
            f = 1.0
        return jnp.float32(LAMBDA_REC * f), jnp.float32(LAMBDA_OVERLAP * f)

    def _train(n_steps, params, opt_state, key, real_imgs, start_step=0):
        history = []
        t0 = _time.time()
        for step in range(start_step, start_step + n_steps):
            key, sk = jr.split(key)
            lam_rec, lam_ov = schedule(step)
            params, opt_state, total, aux = _step(params, opt_state, sk, real_imgs, lam_rec, lam_ov)
            if step % LOG_EVERY == 0 or step == start_step + n_steps - 1:
                l = float(total); n = float(aux[0]); r = float(aux[1]); ov = float(aux[2])
                history.append((step, l, n, r, ov))
                print(f"step {step:5d}  lam_rec={float(lam_rec):6.1f}  total={l:9.2f}  "
                      f"npe={n:9.2f}  rec={r:.4f}  ov={ov:.3f}  "
                      f"({(_time.time()-t0)/(step-start_step+1)*1000:.1f} ms/step)")
        return params, opt_state, key, history

    # Initialise params and optimiser
    init_key = jr.PRNGKey(0)
    params = _init_params(init_key)
    opt_state = _optimizer.init(params)

    # Train (resumable: re-running this cell continues with current params)
    params, opt_state, train_key, history = _train(
        N_STEPS, params, opt_state, jr.PRNGKey(2024), real_tiles,
    )
    print("done")

    return (params,)


@app.cell(hide_code=True)
def _(
    N_SLOTS,
    TILE_H,
    TILE_W,
    encoder_module,
    jax,
    jnp,
    likelihood_module,
    np,
    params,
    plt,
    posterior_module,
    real_tiles,
    simulate_tile,
    unconstrained_to_latents,
):
    # --- Inspect amortized posterior on real tiles ----------------------------

    def _bound_apply_one(module, params, suffix):
        p = {"params": params[f"{suffix}$params"]}
        return lambda x: module.apply(p, x)

    def _bound_apply_lik(params):
        p = {"params": params["likelihood$params"]}
        return lambda bg, comp, off, ls: likelihood_module.apply(p, bg, comp, off, ls)

    @jax.jit
    def infer_latents_mean(params, image):
        encoder = _bound_apply_one(encoder_module, params, "encoder")
        posterior = _bound_apply_one(posterior_module, params, "posterior")
        emb = encoder(image)
        bg_head, slot_head = posterior(emb)
        bg_u = bg_head[:, 0]; slots_u = slot_head[..., 0]
        return unconstrained_to_latents(bg_u, slots_u)

    @jax.jit
    def infer_and_render(params, image):
        latents = infer_latents_mean(params, image)
        likelihood = _bound_apply_lik(params)
        return latents, simulate_tile(likelihood, latents)

    def _show_inference(params, real_tiles, n=6):
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        for i in range(n):
            img = real_tiles[i]
            latents, recon = infer_and_render(params, img)
            latents_np = {k: np.asarray(v) for k, v in latents.items()}
            recon_np = np.clip(np.asarray(recon), 0.0, 1.0)
            axes[i, 0].imshow(np.asarray(img)); axes[i, 0].set_title(f"real {i}")
            axes[i, 1].imshow(recon_np); axes[i, 1].set_title("reconstruction")
            axes[i, 2].imshow(np.asarray(img))
            for s in range(N_SLOTS):
                pres = float(latents_np["presence"][s])
                if pres < 0.3: continue
                cy, cx = latents_np["centers"][s]
                scale = float(jnp.exp(latents_np["log_scale"][s]))
                axes[i, 2].add_patch(plt.Circle((cx, cy), scale, fill=False,
                    edgecolor="red", linewidth=0.5 + 1.5 * pres, alpha=min(1.0, pres)))
            axes[i, 2].set_title(f"inferred (n={int((latents_np['presence']>0.5).sum())})")
            for a in axes[i]: a.set_xticks([]); a.set_yticks([])
            axes[i, 2].set_xlim(0, TILE_W); axes[i, 2].set_ylim(TILE_H, 0)
        fig.tight_layout()
        return fig

    _show_inference(params, real_tiles, n=6)

    return


if __name__ == "__main__":
    app.run()
