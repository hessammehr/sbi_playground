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
    from jax import nn, lax, vmap
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive
    from PIL import Image
    import matplotlib.pyplot as plt
    from functools import partial

    jax.config.update("jax_enable_x64", False)
    print("devices:", jax.devices())
    return (
        Image,
        Predictive,
        dist,
        jax,
        jnp,
        jr,
        lax,
        nn,
        np,
        numpyro,
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
    # Dimensions chosen with 8 GB VRAM in mind. Increase N_SLOTS if you want to
    # resolve more objects per tile; you may need to shrink TILE_* in tandem.

    N_SLOTS = 32          # max objects per tile (presence gates inactive slots)
    K_COMP = 3            # composition dim (interpretable as RGB-ish; mapped via MLP)
    EMB_DIM = 128         # image embedding dim
    LIK_HIDDEN = 64       # hidden width of learned likelihood MLP
    POST_HIDDEN = 128     # hidden width of posterior heads

    # Reasonable scale range: droplets in the example span ~3..30 px radius.
    LOG_SCALE_MIN, LOG_SCALE_MAX = jnp.log(2.0), jnp.log(30.0)

    print(f"slots={N_SLOTS}, K={K_COMP}, tile={TILE_H}x{TILE_W}")

    return (
        EMB_DIM,
        K_COMP,
        LIK_HIDDEN,
        LOG_SCALE_MAX,
        LOG_SCALE_MIN,
        N_SLOTS,
        POST_HIDDEN,
    )


@app.cell(hide_code=True)
def _(EMB_DIM, jnp, jr, lax, nn, real_tiles):
    # --- Small JAX MLP / CNN utilities ----------------------------------------
    # We avoid Flax/Haiku/Equinox -- plain JAX pytrees so numpyro.param can hold
    # the weights directly. Each net exposes (init_params, apply).

    def _glorot(key, shape):
        fan_in, fan_out = shape[0], shape[1]
        lim = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jr.uniform(key, shape, minval=-lim, maxval=lim)

    def mlp_init(key, sizes):
        """sizes = [in, h1, ..., out]"""
        params = []
        keys = jr.split(key, len(sizes) - 1)
        for k, fin, fout in zip(keys, sizes[:-1], sizes[1:]):
            W = _glorot(k, (fin, fout))
            b = jnp.zeros((fout,))
            params.append((W, b))
        return params

    def mlp_apply(params, x, activation=nn.gelu, final_activation=None):
        for i, (W, b) in enumerate(params):
            x = x @ W + b
            if i < len(params) - 1:
                x = activation(x)
        if final_activation is not None:
            x = final_activation(x)
        return x

    # --- CNN embedding -------------------------------------------------
    # A tiny conv stack: 3 -> 16 -> 32 -> 64 (stride-2 each) then global mean pool.
    # Input: (H, W, 3). Output: (EMB_DIM,).

    def conv_init(key, in_c, out_c, k=3):
        W = _glorot(key, (k * k * in_c, out_c)).reshape(k, k, in_c, out_c)
        b = jnp.zeros((out_c,))
        return (W, b)

    def conv2d_stride2(x, W, b):
        # x: (H, W, C). Use jax.lax.conv_general_dilated.
        x_nchw = jnp.transpose(x, (2, 0, 1))[None]            # (1, C, H, W)
        Wt = jnp.transpose(W, (3, 2, 0, 1))                    # (out, in, kH, kW)
        y = lax.conv_general_dilated(
            x_nchw, Wt, window_strides=(2, 2), padding="SAME"
        )
        y = jnp.transpose(y[0], (1, 2, 0)) + b                 # (H/2, W/2, out)
        return y

    def cnn_init(key, emb_dim=EMB_DIM):
        k1, k2, k3, k4 = jr.split(key, 4)
        return dict(
            c1=conv_init(k1, 3, 16),
            c2=conv_init(k2, 16, 32),
            c3=conv_init(k3, 32, 64),
            head=mlp_init(k4, [64, emb_dim]),
        )

    def cnn_apply(params, x):
        """x: (H, W, 3) -> (emb_dim,)"""
        x = nn.gelu(conv2d_stride2(x, *params["c1"]))
        x = nn.gelu(conv2d_stride2(x, *params["c2"]))
        x = nn.gelu(conv2d_stride2(x, *params["c3"]))
        x = x.mean(axis=(0, 1))    # global avg pool -> (64,)
        x = mlp_apply(params["head"], x)
        return x

    # quick shape test
    _k = jr.PRNGKey(0)
    _p = cnn_init(_k)
    _e = cnn_apply(_p, real_tiles[0])
    print("embedding shape:", _e.shape)

    return cnn_apply, cnn_init, mlp_apply, mlp_init


@app.cell(hide_code=True)
def _(K_COMP, LIK_HIDDEN, jnp, jr, mlp_apply, mlp_init, np):
    # --- Learned per-pixel likelihood ----------------------------------------
    #
    # Models how a single droplet modifies the background at a single pixel.
    # Inputs per (droplet, pixel):
    #   bg_color    : (3,)   current colour at that pixel before this droplet
    #   comp        : (K,)   composition latent (mapped to a colour change by the MLP)
    #   offset_norm : (2,)   (px - center) mapped through droplet shape matrix L
    #   log_scale   : ()
    #   presence    : ()
    #
    # Output: new colour, with both invariants exact:
    #   out = bg + presence * envelope(||offset_norm||) * delta(...)
    # envelope(r) = exp(-0.5 r^2) -> 0 as r -> infinity.
    #
    # The shape is non-radial: L is built from (log_scale, log_aspect, theta) so
    # the network itself never has to know about ellipticity.

    LIK_IN = 3 + K_COMP + 2 + 1

    def lik_init(key, hidden=LIK_HIDDEN):
        params = mlp_init(key, [LIK_IN, hidden, hidden, 3])
        # Shrink the final layer to keep initial deltas small (avoid saturating
        # the prior predictive). The network will learn larger deltas as needed.
        W, b = params[-1]
        params[-1] = (W * 0.3, b)
        return params

    def lik_apply(lik_params, bg, comp, offset_norm, log_scale, presence):
        feats = jnp.concatenate([
            bg, comp, offset_norm,
            log_scale[..., None] if jnp.ndim(log_scale) > 0 else jnp.array([log_scale]),
        ], axis=-1)
        delta = mlp_apply(lik_params, feats)
        r2 = (offset_norm ** 2).sum(axis=-1)
        env = jnp.exp(-0.5 * r2)
        gate = (presence * env)[..., None]
        return bg + gate * delta

    # sanity check
    _lp = lik_init(jr.PRNGKey(1))
    _out_c = lik_apply(_lp, bg=jnp.array([0.5,0.5,0.5]), comp=jnp.zeros(K_COMP),
                       offset_norm=jnp.array([0.,0.]), log_scale=jnp.array(2.0),
                       presence=jnp.array(1.0))
    _out_far = lik_apply(_lp, bg=jnp.array([0.5,0.5,0.5]), comp=jnp.zeros(K_COMP),
                         offset_norm=jnp.array([100.,100.]), log_scale=jnp.array(2.0),
                         presence=jnp.array(1.0))
    _out_zp = lik_apply(_lp, bg=jnp.array([0.5,0.5,0.5]), comp=jnp.array([1.,2.,3.]),
                        offset_norm=jnp.array([0.,0.]), log_scale=jnp.array(2.0),
                        presence=jnp.array(0.0))
    print("center:", np.asarray(_out_c).round(3),
          "far:", np.asarray(_out_far).round(3),
          "zero-pres:", np.asarray(_out_zp).round(3))

    return (lik_init,)


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
    jr,
    lax,
    mlp_apply,
    numpyro,
):
    # --- Background prior + droplet shape parameterization --------------------

    # --- Overlap penalty (pairwise Gaussian repulsion) ------------------------
    # Differentiable, presence-gated soft-DPP / Strauss-style pair potential:
    #
    #   E_overlap(X) = sum_{i<j} pres_i * pres_j * exp(-||c_i - c_j||^2 / (2 r^2))
    #
    # Used as (i) a numpyro.factor inside prior_model so the joint prior is
    # repulsive, and (ii) an explicit regulariser in the training loss that the
    # posterior also pays at inference time (keeps slots from clustering).

    OVERLAP_R = 6.0   # repulsion radius in pixels; ~min droplet radius
    OVERLAP_W = 4.0   # log-weight; tune so E[OVERLAP_W * E_overlap] is comparable
                      # to the other prior terms.

    def overlap_penalty(centers, presence, r=OVERLAP_R):
        diff = centers[:, None, :] - centers[None, :, :]
        d2 = (diff ** 2).sum(-1)                                # (N, N)
        pair = jnp.exp(-d2 / (2.0 * r ** 2))                    # (N, N)
        mask = presence[:, None] * presence[None, :]            # (N, N)
        # Strict upper triangle so each pair counted once and diagonal is excluded
        triu = jnp.triu(jnp.ones_like(pair), k=1)
        return (pair * mask * triu).sum()

    def render_background(bg_corners, H, W):
        """bg_corners: (4, 3) for TL, TR, BL, BR. Bilinear interp -> (H, W, 3)."""
        ys = jnp.linspace(0.0, 1.0, H)
        xs = jnp.linspace(0.0, 1.0, W)
        Y, X = jnp.meshgrid(ys, xs, indexing="ij")
        tl, tr, bl, br = bg_corners[0], bg_corners[1], bg_corners[2], bg_corners[3]
        top = (1 - X)[..., None] * tl + X[..., None] * tr
        bot = (1 - X)[..., None] * bl + X[..., None] * br
        return (1 - Y)[..., None] * top + Y[..., None] * bot

    def droplet_L(log_scale, log_aspect, theta):
        s = jnp.exp(log_scale)
        a = jnp.exp(log_aspect)
        sx, sy = s * a, s / a
        c, sn = jnp.cos(theta), jnp.sin(theta)
        R = jnp.array([[c, -sn], [sn, c]])
        D = jnp.diag(jnp.array([1.0 / sx, 1.0 / sy]))
        return D @ R.T

    def simulate_tile(lik_params, latents, H=TILE_H, W=TILE_W):
        """Composite all droplets onto background. Both invariants preserved:
           presence=0 contributes nothing; envelope decays to 0 at infinity.
           We soft-saturate at [0,1] via clipping at the very end (post-loop) so
           the embedding net never sees explosive values."""
        bg = render_background(latents["bg_corners"], H, W)
        ys = jnp.arange(H, dtype=jnp.float32)
        xs = jnp.arange(W, dtype=jnp.float32)
        Y, X = jnp.meshgrid(ys, xs, indexing="ij")
        px = jnp.stack([Y, X], axis=-1)

        def add_droplet(img, slot):
            center, ls, la, th, comp, pres = slot
            L = droplet_L(ls, la, th)
            rel = px - center
            offset_norm = rel @ L.T
            r2 = (offset_norm ** 2).sum(-1)
            env = jnp.exp(-0.5 * r2)
            feats = jnp.concatenate([
                img,
                jnp.broadcast_to(comp, (H, W, K_COMP)),
                offset_norm,
                jnp.broadcast_to(ls, (H, W))[..., None],
            ], axis=-1)
            delta = mlp_apply(lik_params, feats)
            new = img + (pres * env)[..., None] * delta
            return new, None

        slots = (
            latents["centers"],
            latents["log_scale"],
            latents["log_aspect"],
            latents["theta"],
            latents["comp"],
            latents["presence"],
        )
        # checkpoint the scan body so reverse-mode AD doesn't store all N_SLOTS iterates
        img, _ = lax.scan(jax.checkpoint(add_droplet), bg, slots)
        return img  # unclipped; clip only for display

    # --- Prior model ------------------------------------------------------------
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
        # Repulsion factor on the joint prior (presence-gated).
        numpyro.factor("overlap", -OVERLAP_W * overlap_penalty(centers, presence))
        return dict(
            bg_corners=bg_corners, centers=centers,
            log_scale=log_scale, log_aspect=log_aspect, theta=theta,
            comp=comp, presence=presence,
        )

    # Helper: turn a flat Predictive sample dict into the packed latents dict.
    def pack_latents(samples):
        """samples may have a leading batch dim or not. Works for both."""
        return dict(
            bg_corners=samples["bg_corners"],
            centers=jnp.stack([samples["cy"], samples["cx"]], axis=-1),
            log_scale=samples["log_scale"],
            log_aspect=samples["log_aspect"],
            theta=samples["theta"],
            comp=samples["comp"],
            presence=samples["presence"],
        )

    OBS_SIGMA = 0.03  # pixel observation noise

    def add_obs_noise(key, img, sigma=OBS_SIGMA):
        return jnp.clip(img + sigma * jr.normal(key, img.shape), 0.0, 1.0)

    print("simulator + prior defined")

    return OBS_SIGMA, overlap_penalty, pack_latents, prior_model, simulate_tile


@app.cell(hide_code=True)
def _(Predictive, jax, jnp, jr, lik_init, np, plt, prior_model, simulate_tile):
    # --- Prior predictive check ------------------------------------------------
    # Sample latents from the prior via numpyro Predictive, then run them through
    # the simulator with a RANDOMLY-INITIALISED likelihood net to see what kinds of
    # images the model can produce. We expect a wide range of droplet locations,
    # sizes, shapes, and colours (broad coverage, per arXiv:2310.04395).
    import time

    lik_params_init = lik_init(jr.PRNGKey(42))

    @jax.jit
    def _sample_and_sim(key, lik_params):
        pred = Predictive(prior_model, num_samples=1)
        samples = pred(key)
        latents = {k: v[0] for k, v in samples.items()}
        latents_packed = dict(
            bg_corners=latents["bg_corners"],
            centers=jnp.stack([latents["cy"], latents["cx"]], axis=-1),
            log_scale=latents["log_scale"],
            log_aspect=latents["log_aspect"],
            theta=latents["theta"],
            comp=latents["comp"],
            presence=latents["presence"],
        )
        img = simulate_tile(lik_params, latents_packed)
        return img, latents_packed

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
        fig.suptitle("Prior predictive (random likelihood net) -- 12 samples")
        fig.tight_layout()
        return fig
    _do_ppc()

    return


@app.cell(hide_code=True)
def _(
    EMB_DIM,
    K_COMP,
    LOG_SCALE_MAX,
    LOG_SCALE_MIN,
    N_SLOTS,
    POST_HIDDEN,
    TILE_H,
    TILE_W,
    jax,
    jnp,
    jr,
    mlp_apply,
    mlp_init,
):
    # --- Amortized posterior network ------------------------------------------
    #
    # q_eta(theta | x) factorizes over: 4 bg-corner means + per-slot heads. The
    # image is summarized by the CNN embedding (EMB_DIM,). All distributions are
    # Normal in unconstrained space; we map to the constrained latent spaces with
    # the same transforms numpyro would (Uniform <-> sigmoid; presence Beta <->
    # sigmoid of a logit). Sampling uses the reparameterization trick.
    #
    # Per-slot output (unconstrained):
    #   cy_u, cx_u     -> Uniform(0,H/W)   via sigmoid * H/W
    #   log_scale_u    -> Uniform(LS_MIN, LS_MAX) via sigmoid + scale
    #   log_aspect     -> Normal              identity
    #   theta_u        -> Uniform(-pi/2, pi/2) via sigmoid * pi - pi/2
    #   comp           -> Normal              identity (K dims)
    #   presence_u     -> Beta(0.3,0.3)       via sigmoid -- approx by Normal in
    #                                         logit space (we sample logit then
    #                                         sigmoid; the variational family is
    #                                         Logistic-Normal which approximates
    #                                         the bimodal Beta well enough).
    # All heads predict (mu, log_sigma) per scalar (and per-dim for comp).

    # Number of unconstrained scalars per slot:
    PER_SLOT_DIMS = 2 + 1 + 1 + 1 + K_COMP + 1   # cy,cx, ls, la, th, comp(K), pres
    BG_DIMS = 4 * 3  # 12

    def post_init(key, emb_dim=EMB_DIM, hidden=POST_HIDDEN):
        k_bg, k_slot = jr.split(key)
        return dict(
            bg_head=mlp_init(k_bg, [emb_dim, hidden, 2 * BG_DIMS]),
            slot_head=mlp_init(k_slot, [emb_dim, hidden, N_SLOTS * 2 * PER_SLOT_DIMS]),
        )

    def post_apply(post_params, embedding):
        bg = mlp_apply(post_params["bg_head"], embedding).reshape(BG_DIMS, 2)
        slots = mlp_apply(post_params["slot_head"], embedding).reshape(N_SLOTS, PER_SLOT_DIMS, 2)
        return bg, slots   # each last-axis = (mu, log_sigma)

    # --- Sample from posterior, return both packed latents and the unconstrained
    # vector + log_q (entropy-like; for NPE training we will use the log-prob of
    # the *true* unconstrained latents under q).

    def _logit(x, eps=1e-6):
        x = jnp.clip(x, eps, 1.0 - eps)
        return jnp.log(x) - jnp.log1p(-x)

    def true_latents_to_unconstrained(latents):
        """Map a packed-latents dict (constrained) to a (BG_DIMS,) + (N, PER_SLOT_DIMS)
        pair of unconstrained vectors that match the variational family."""
        bg = latents["bg_corners"].reshape(BG_DIMS)               # identity (Normal)
        cy = _logit(latents["centers"][:, 0] / TILE_H)
        cx = _logit(latents["centers"][:, 1] / TILE_W)
        ls = _logit((latents["log_scale"] - LOG_SCALE_MIN) / (LOG_SCALE_MAX - LOG_SCALE_MIN))
        la = latents["log_aspect"]
        th = _logit((latents["theta"] + jnp.pi/2) / jnp.pi)
        comp = latents["comp"]                                    # (N, K)
        pres = _logit(latents["presence"])
        per_slot = jnp.concatenate([
            cy[:, None], cx[:, None], ls[:, None], la[:, None],
            th[:, None], comp, pres[:, None],
        ], axis=-1)                                               # (N, PER_SLOT_DIMS)
        return bg, per_slot

    def unconstrained_to_latents(bg_u, slots_u):
        """Inverse of the above (deterministic transforms only)."""
        bg_corners = bg_u.reshape(4, 3)
        cy = jax.nn.sigmoid(slots_u[..., 0]) * TILE_H
        cx = jax.nn.sigmoid(slots_u[..., 1]) * TILE_W
        ls = LOG_SCALE_MIN + jax.nn.sigmoid(slots_u[..., 2]) * (LOG_SCALE_MAX - LOG_SCALE_MIN)
        la = slots_u[..., 3]
        th = jax.nn.sigmoid(slots_u[..., 4]) * jnp.pi - jnp.pi/2
        comp = slots_u[..., 5:5+K_COMP]
        pres = jax.nn.sigmoid(slots_u[..., 5+K_COMP])
        centers = jnp.stack([cy, cx], axis=-1)
        return dict(
            bg_corners=bg_corners, centers=centers,
            log_scale=ls, log_aspect=la, theta=th, comp=comp, presence=pres,
        )

    def gaussian_logprob(x, mu, log_sigma):
        sigma = jnp.exp(log_sigma)
        return -0.5 * ((x - mu) / sigma) ** 2 - log_sigma - 0.5 * jnp.log(2 * jnp.pi)

    LOG_SIGMA_CLAMP = (-4.0, 2.0)

    def post_log_prob(post_params, embedding, bg_u, slots_u, slot_mask=None):
        """log q(theta_u | x). If slot_mask is given (shape (N,)), only those slots
        contribute to the slot term. Background term always contributes."""
        bg_head, slot_head = post_apply(post_params, embedding)
        mu_b, ls_b = bg_head[:, 0], jnp.clip(bg_head[:, 1], *LOG_SIGMA_CLAMP)
        mu_s, ls_s = slot_head[..., 0], jnp.clip(slot_head[..., 1], *LOG_SIGMA_CLAMP)
        lp_bg = gaussian_logprob(bg_u, mu_b, ls_b).sum()
        lp_slot_full = gaussian_logprob(slots_u, mu_s, ls_s).sum(-1)   # (N,)
        if slot_mask is None:
            lp_slot = lp_slot_full.sum()
        else:
            lp_slot = (lp_slot_full * slot_mask).sum()
        return lp_bg + lp_slot

    def slots_u_targets_split(slot_head):
        # slot_head: (N, PER_SLOT_DIMS, 2)
        return slot_head[..., 0], slot_head[..., 1]

    def post_sample(post_params, embedding, key):
        """Reparameterized sample of (bg_u, slots_u) from q."""
        bg_head, slot_head = post_apply(post_params, embedding)
        mu_b, ls_b = bg_head[:, 0], jnp.clip(bg_head[:, 1], *LOG_SIGMA_CLAMP)
        mu_s, ls_s = slot_head[..., 0], jnp.clip(slot_head[..., 1], *LOG_SIGMA_CLAMP)
        k1, k2 = jr.split(key)
        bg_u = mu_b + jnp.exp(ls_b) * jr.normal(k1, mu_b.shape)
        slots_u = mu_s + jnp.exp(ls_s) * jr.normal(k2, mu_s.shape)
        return bg_u, slots_u

    # Canonical ordering: sort slots so that present ones come first (presence>0.5),
    # within each group sorted by center_y then center_x. This breaks slot
    # permutation symmetry deterministically during training.

    def canonical_order(latents):
        present = (latents["presence"] > 0.5).astype(jnp.float32)
        # primary key: -present (present first), secondary cy, tertiary cx
        cy = latents["centers"][:, 0]
        cx = latents["centers"][:, 1]
        # Compose a single sort key: present rank * big_offset + cy * H + cx
        H = float(TILE_H)
        W = float(TILE_W)
        key = (1.0 - present) * 1e8 + cy * (W + 10.0) + cx
        order = jnp.argsort(key)
        def reorder(arr):
            return arr[order]
        return dict(
            bg_corners=latents["bg_corners"],
            centers=reorder(latents["centers"]),
            log_scale=reorder(latents["log_scale"]),
            log_aspect=reorder(latents["log_aspect"]),
            theta=reorder(latents["theta"]),
            comp=reorder(latents["comp"]),
            presence=reorder(latents["presence"]),
        )

    def post_log_prob_set(post_params, embedding, bg_u, slots_u, present_mask):
        """Permutation-invariant NPE log-prob (DETR-style soft Hungarian).

        Args:
          bg_u: (BG_DIMS,)            -- target background (unconstrained)
          slots_u: (N_true, PER_SLOT_DIMS) -- target slot latents in canonical order
          present_mask: (N_true,)     -- which canonical slots are PRESENT (0/1)

        Treats predicted heads as a uniform mixture over slots; each TRUE slot
        is scored by the logsumexp over predicted heads, weighted by its mask.
        Background term is unchanged.
        """
        bg_head, slot_head = post_apply(post_params, embedding)
        mu_b, ls_b = bg_head[:, 0], jnp.clip(bg_head[:, 1], *LOG_SIGMA_CLAMP)
        mu_s, ls_s = slot_head[..., 0], jnp.clip(slot_head[..., 1], *LOG_SIGMA_CLAMP)
        # bg term
        lp_bg = gaussian_logprob(bg_u, mu_b, ls_b).sum()
        # pairwise log-prob: for each true slot i and each predicted slot j,
        # log N(true_u[i] | mu_s[j], sigma_s[j]) summed over latent dims.
        # shape: (N_true, N_pred)
        diff = slots_u[:, None, :] - mu_s[None, :, :]                # (Nt, Np, D)
        z2 = (diff / jnp.exp(ls_s)[None, :, :]) ** 2                 # (Nt, Np, D)
        lp_pair = (-0.5 * z2 - ls_s[None, :, :] - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
        # mixture log-prob per true slot: logsumexp over predicted slots - log Np
        Np = mu_s.shape[0]
        lp_per_true = jax.scipy.special.logsumexp(lp_pair, axis=1) - jnp.log(Np)  # (Nt,)
        lp_slot = (lp_per_true * present_mask).sum()
        return lp_bg + lp_slot

    print("posterior net defined; per-slot dims:", PER_SLOT_DIMS)

    return (
        post_apply,
        post_init,
        post_log_prob_set,
        true_latents_to_unconstrained,
        unconstrained_to_latents,
    )


@app.cell(hide_code=True)
def _(
    OBS_SIGMA,
    Predictive,
    cnn_apply,
    cnn_init,
    jax,
    jnp,
    jr,
    lik_init,
    overlap_penalty,
    pack_latents,
    post_apply,
    post_init,
    post_log_prob_set,
    prior_model,
    simulate_tile,
    true_latents_to_unconstrained,
    unconstrained_to_latents,
    vmap,
):
    # --- Training: amortized NPE + simulator self-consistency ----------------
    #
    # Joint loss per minibatch step:
    #
    #   L_npe  =  E_{theta ~ prior, x = clip(sim(theta; phi) + noise)}
    #                [ -log q(theta_canon | x) ]      (only over PRESENT slots)
    #   L_rec  =  E_{x_real}  || sim( decode( posterior_mean( x_real ) ); phi ) - x_real ||^2
    #   L      =  L_npe + lambda_rec * L_rec
    #
    # Notes
    # - We only score the NPE log-prob on slots that are PRESENT in the simulated
    #   sample (presence>0.5 after canonical ordering puts them first). Absent
    #   slots have arbitrary latents from the prior; scoring them would just add
    #   noise.
    # - Reconstruction uses the posterior MEAN (no sampling). Cleaner gradients.

    import optax

    BATCH = 8
    LAMBDA_REC = 5000.0     # rescale so both terms have comparable gradient magnitudes
    LR = 3e-4

    def init_all_params(key):
        k1, k2, k3 = jr.split(key, 3)
        return dict(cnn=cnn_init(k1), post=post_init(k2), lik=lik_init(k3))

    def sim_batch(key, lik_params, B):
        pred = Predictive(prior_model, num_samples=B)
        samples = pred(key)
        packed_batch = pack_latents(samples)
        def one(i):
            single = jax.tree_util.tree_map(lambda v: v[i], packed_batch)
            return simulate_tile(lik_params, single)
        imgs = vmap(one)(jnp.arange(B))
        nk = jr.fold_in(key, 1)
        imgs = jnp.clip(imgs + OBS_SIGMA * jr.normal(nk, imgs.shape), 0.0, 1.0)
        return packed_batch, imgs

    def npe_loss_one(cnn_p, post_p, latents, image):
        """Permutation-invariant NPE loss (DETR-style soft Hungarian).
        Each true present droplet is scored by logsumexp over predicted slot heads."""
        emb = cnn_apply(cnn_p, image)
        bg_u, slots_u = true_latents_to_unconstrained(latents)
        present_mask = (latents["presence"] > 0.5).astype(jnp.float32)
        return -post_log_prob_set(post_p, emb, bg_u, slots_u, present_mask)

    def recon_loss_one(cnn_p, post_p, lik_p, image):
        """Use posterior MEAN to render (no sampling) -- direct supervision on
        the deterministic forward map. Returns (mse, latents) so the caller can
        also penalise overlap at inference time."""
        emb = cnn_apply(cnn_p, image)
        bg_head, slot_head = post_apply(post_p, emb)
        bg_u = bg_head[:, 0]
        slots_u = slot_head[..., 0]
        latents = unconstrained_to_latents(bg_u, slots_u)
        sim = simulate_tile(lik_p, latents)
        mse = jnp.mean((sim - image) ** 2)
        return mse, latents

    LAMBDA_OVERLAP = 50.0  # pressure on inferred posterior centers to spread out

    def total_loss(p, sim_latents, sim_images, real_images):
        def per_npe(i, img):
            latents_i = jax.tree_util.tree_map(lambda v: v[i], sim_latents)
            return npe_loss_one(p["cnn"], p["post"], latents_i, img)
        L_npe = vmap(per_npe)(jnp.arange(sim_images.shape[0]), sim_images).mean()
        def per_rec(img):
            mse, lat = recon_loss_one(p["cnn"], p["post"], p["lik"], img)
            ov = overlap_penalty(lat["centers"], lat["presence"])
            return mse, ov
        rec_mses, rec_ovs = vmap(per_rec)(real_images)
        L_rec = rec_mses.mean()
        L_ov = rec_ovs.mean()
        total = L_npe + LAMBDA_REC * L_rec + LAMBDA_OVERLAP * L_ov
        return total, (L_npe, L_rec, L_ov)

    optimizer = optax.adam(LR)

    @jax.jit
    def train_step(params, opt_state, key, real_images):
        sim_latents, sim_images = sim_batch(key, params["lik"], BATCH)
        (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
            params, sim_latents, sim_images, real_images,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    params_init = init_all_params(jr.PRNGKey(0))
    opt_state_init = optimizer.init(params_init)

    def _count(p):
        return sum(x.size for x in jax.tree_util.tree_leaves(p))
    print("param counts -- cnn:", _count(params_init["cnn"]),
          " post:", _count(params_init["post"]),
          " lik:", _count(params_init["lik"]))

    return opt_state_init, params_init, train_step


@app.cell(hide_code=True)
def _(jr, opt_state_init, params_init, real_tiles, train_step):
    # --- Training loop (resumable) -------------------------------------------
    # Owns `params` and `opt_state`. Re-running this cell continues training from
    # the current values; reset by editing the seeding below or rerunning the cell
    # above that defines `params_init`.

    import time as _time

    N_STEPS = 1000
    LOG_EVERY = 100

    def _train_for(n_steps, params, opt_state, key, real_images):
        history = []
        t0 = _time.time()
        for step in range(n_steps):
            key, sk = jr.split(key)
            params, opt_state, loss, aux = train_step(params, opt_state, sk, real_images)
            if step % LOG_EVERY == 0 or step == n_steps - 1:
                l = float(loss); n = float(aux[0]); r = float(aux[1]); ov = float(aux[2])
                history.append((step, l, n, r, ov))
                print(f"step {step:5d}  total={l:9.2f}  npe={n:9.2f}  rec={r:.4f}  ov={ov:.3f}  "
                      f"({(_time.time()-t0)/(step+1)*1000:.1f} ms/step)")
        return params, opt_state, key, history

    # Run training. First call starts from params_init / opt_state_init.
    params, opt_state, train_key, history = _train_for(
        N_STEPS, params_init, opt_state_init, jr.PRNGKey(2024), real_tiles,
    )
    print("done")

    return (params,)


@app.cell(hide_code=True)
def _(
    N_SLOTS,
    TILE_H,
    TILE_W,
    cnn_apply,
    jax,
    jnp,
    np,
    params,
    plt,
    post_apply,
    real_tiles,
    simulate_tile,
    unconstrained_to_latents,
):
    # --- Inspect amortized posterior on real tiles ----------------------------
    # Forward pass only -- this is the deliverable: a single image -> latents.

    @jax.jit
    def infer_latents_mean(params, image):
        """Return the posterior-mean latents (point estimate) for one image."""
        emb = cnn_apply(params["cnn"], image)
        bg_head, slot_head = post_apply(params["post"], emb)
        bg_u = bg_head[:, 0]
        slots_u = slot_head[..., 0]
        return unconstrained_to_latents(bg_u, slots_u)

    @jax.jit
    def infer_and_render(params, image):
        latents = infer_latents_mean(params, image)
        recon = simulate_tile(params["lik"], latents)
        return latents, recon

    def _show_inference(params, real_tiles, n=6):
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        for i in range(n):
            img = real_tiles[i]
            latents, recon = infer_and_render(params, img)
            latents_np = {k: np.asarray(v) for k, v in latents.items()}
            recon_np = np.asarray(recon)
            # column 0: real
            axes[i, 0].imshow(np.asarray(img)); axes[i, 0].set_title(f"real {i}")
            # column 1: reconstruction
            axes[i, 1].imshow(recon_np); axes[i, 1].set_title("reconstruction")
            # column 2: real + droplet markers
            axes[i, 2].imshow(np.asarray(img))
            for s in range(N_SLOTS):
                pres = float(latents_np["presence"][s])
                if pres < 0.3:
                    continue
                cy, cx = latents_np["centers"][s]
                scale = float(jnp.exp(latents_np["log_scale"][s]))
                axes[i, 2].add_patch(plt.Circle(
                    (cx, cy), scale, fill=False,
                    edgecolor="red", linewidth=0.5 + 1.5 * pres, alpha=min(1.0, pres)
                ))
            axes[i, 2].set_title("inferred droplets")
            for a in axes[i]: a.set_xticks([]); a.set_yticks([])
            axes[i, 2].set_xlim(0, TILE_W); axes[i, 2].set_ylim(TILE_H, 0)
        fig.tight_layout()
        return fig

    _show_inference(params, real_tiles, n=6)

    return


if __name__ == "__main__":
    app.run()
