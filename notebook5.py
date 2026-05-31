# /// script
# dependencies = [
#     "jax[cpu,cuda12]==0.10.1",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "numpyro==0.21.0",
#     "optax==0.2.8",
#     "pillow==12.2.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Simulation-Based Amortised Bayesian Inference for Microscopy Droplet Patches

        **Goal.** Infer *structured, interpretable* object latents (presence/count,
        position, size, composition) from a handful of **unlabelled** real microscopy
        images (`example.jpg`: dense, colourful thin-film-interference droplets) via a
        coherent NumPyro probabilistic model, simulation-based pretraining (NPE), and
        Bayesian self-consistency adaptation. **Not** image autoencoding / reconstruction.

        Self-contained restart distilling `notebook4.py`. Shared infrastructure is defined
        once; four model **generations** (`v1`..`v4`) are fully implemented and selectable
        via the dropdown in section 7. Each is a `ModelVersion` bundle that plugs into the
        shared NPE / self-consistency / diagnostics machinery.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Background, principles, and the ABI/SBI workflow

        **Workflow.** (1) define prior `p(theta)` + simulator `p(x|theta)`; (2) sample
        labelled synthetic pairs `(theta,x)`; (3) train an amortised posterior
        `q_phi(theta|x)` by NPE, minimising `L_NPE = -log q_phi(theta_sim|x_sim)`;
        (4) evaluate posterior / posterior-predictive diagnostics; (5) **adapt** to
        unlabelled real images by **Bayesian self-consistency**.

        **Self-consistency objective (real images).** For real `x`, draw proposals from a
        *stop-gradient* current guide `theta_l ~ q_phi_old(theta|x)` and minimise

        $$ L_\text{SC} = \operatorname{Var}_l\big[\, \log p_\text{model}(\theta_l, x) - \log q_\phi(\theta_l\mid x) \,\big]. $$

        At the true amortised posterior the log-ratio is constant in `theta` (= `log p(x)`),
        so zero variance ⇔ consistency. This is **not** reconstruction MSE, **not** cycle
        consistency. Any unknown image-calibration must be **explicit low-dim model
        variables**, not silent preprocessing.

        **Interpretability constraint (hard).** Each object = {presence, position(2D),
        size(>0), composition(low-dim constrained, e.g. simplex)}. No image autoencoder, no
        generic slot latent, no VAE bottleneck, no arbitrary per-image embedding. A learned
        *renderer* is allowed but must be **global/shared** (no per-image latent, no
        per-object embedding) and conditioned only on interpretable object state + local
        pixel/background context.

        **Literature touched (conceptual; implementation is NumPyro-native).**
        BayesFlow (Radev et al. 2020+) — amortised SBI design patterns (NOT a dependency);
        NPE/SNPE (Papamakarios & Murray 2016; Greenberg et al. 2019); the SBI frontier
        (Cranmer, Brehmer, Louppe 2020, PNAS); self-consistency / ratio-consistency for
        amortised posteriors; object-centric rendering (AIR, Eslami et al. 2016; slot
        attention) — conceptual only, we keep latents interpretable; coordinate/implicit
        neural rendering (NeRF; random Fourier features, Tancik et al. 2020) for the learned
        per-object sprite.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Lessons learned (respect these)

        **Identifiability.** Label-switching is fatal for NPE on order-invariant renderers
        (same `x` ↦ many permuted `theta`). Fix: **anchor objects to a spatial grid**, each
        cell holding ≤1 object placed uniformly *within* its cell — identical free-placement
        prior, but a canonical identifiable labelling. A scaffold, not physics.

        **Guide architecture (biggest bug).** CNN → *flatten → global FC* destroys per-object
        spatial info: position survives (where features fire) but **colour/composition
        collapses to the dataset mean**. Fix: **slot-aligned guide** — conv stack →
        feature map at grid resolution → shared **per-cell MLP** reading features *at each
        cell* + a global-context vector. Recovered colour std-ratio 0.25 → ~0.9. Still a
        valid NumPyro `q_phi(theta|x)`; architecture change, not a trick.

        **Evaluation discipline.** Never gate on reconstruction/image-MSE (it ranks *empty*
        patches as best). Gate on **ground-truth latent recovery** on **non-empty** scenes,
        stratified by count, vs a **prior-mean baseline**. Always include train-set sanity /
        tiny-overfit. Visualise best, worst, and failure cases — not flattering averages.

        **Real data (`example.jpg`).** RGB, dense, bright background, specular highlights,
        thin-film fringes. Use **native 64×64 crops, no resize/downsample, no colour
        normalisation** (downsampling crams ~100 droplets per patch ≫ capacity; native crops
        give ~3–12). The v3 SC fit to real patches was the key diagnostic: finite &
        differentiable, but the **model** (not inference) is the bottleneck — render std
        ~30–40% of real, `SIZE_HIGH` too small for big droplets, single smooth sprite can't
        make rings/highlights/fringes, flat background, near-fixed colour.

        **Renderer flexibility (v4 frontier).** Colours looked *fixed even before training*
        because composition entered a fixed linear palette matrix (v3). v4 removes this:
        fully-learned composition→appearance (random-Fourier coord MLP, no palette), wider
        learned size, free placement, higher cardinality, graded background — no hard-coded
        optics. Caveat: an *untrained* flexible renderer makes weak ambiguous images, so
        init it for visible/localised/colour-expressive objects (freely overridable by
        training). v4 status: pos/size/colour recover well; **count/presence is the weak
        point** under free placement.

        **SC hygiene.** Tiny LR + grad clip; large SC steps wreck the synthetic posterior.
        Monitor a fixed-proposal SC value and synthetic held-out metrics for collapse.

        **Compute.** Rendering `S×B×K×H×W` OOMs; **chunk** renders/log-probs; don't `vmap`
        a full-scene render over thousands of items. Marimo cascades re-run downstream
        (incl. long training) — write big edits to temp files; `scp` saved PNGs to inspect.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Status & roadmap

        | stage | v1 toy grayscale 32² | v2 grayscale anchored | v3 RGB fixed-palette 64² | v4 fully-learned 64² |
        |---|---|---|---|---|
        | prior + renderer | ✅ Gaussian blobs | ✅ + x-anchored slots | ✅ coord-MLP, comp→RGB matrix | ✅ Fourier MLP, free place, wide size |
        | slot-aligned guide | ✅ (MLP) | ✅ (CNN) | ✅ 16-slot | ✅ 64-cell (8×8) |
        | synthetic NPE | ✅ | ✅ | ✅ colour ratio ~0.9 | ⚠️ pos/size/colour good, count weak |
        | real-image SC | ✅ pseudo-real | — | ✅ diagnostic (model-limited) | ⟳ next |

        **Next for a new session.** (a) reproduce v4 training here; (b) improve
        **count/presence** under free placement (count-aware loss / learned presence prior /
        visibility-weighted cells), keep `q_phi` proper; (c) real-image SC with trained v4,
        `scp` obs|fit|residual and compare to v3; (d) only then add controlled extensions
        (PSF/blur, camera gain/offset, anisotropy, non-overlap prior), one named mechanism
        at a time, re-running diagnostics.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Imports & global config""")
    return


@app.cell
def _():
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from dataclasses import dataclass, field
    from typing import Callable

    import numpy as np
    import matplotlib.pyplot as plt

    import jax
    import jax.numpy as jnp
    import jax.nn as jnn
    from jax import random

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive
    from numpyro.infer.util import log_density
    from numpyro import handlers

    import optax

    print("jax", jax.__version__, "devices", jax.devices())
    return (
        Callable,
        Predictive,
        dataclass,
        dist,
        field,
        handlers,
        jax,
        jnn,
        jnp,
        log_density,
        np,
        numpyro,
        optax,
        plt,
        random,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Shared infrastructure (defined once)

        A **`ModelVersion`** exposes a uniform interface so the shared NPE / self-consistency
        / diagnostics code drives every generation identically. All versions use the *generic*
        observed-site name **`"obs"`** and deterministic **`"mean"`/`"count"`**, and a tuple
        of latent `site_names`; per-version specifics live inside the closures.
        """
    )
    return


@app.cell
def _(Callable, dataclass, field):
    @dataclass
    class ModelVersion:
        name: str
        description: str
        image_shape: tuple
        channels: int
        max_objects: int
        composition_dim: int
        site_names: tuple                 # latent sample-site names
        position_low: object              # (max_objects, 2)
        position_high: object
        size_low: float
        size_high: float
        model: Callable                   # model(image=None) -> NumPyro model, sites: site_names + "mean","count","obs"
        init_guide_params: Callable       # (key) -> pytree
        guide: Callable                   # (image, guide_params) NumPyro guide, mirrored sites
        guide_log_prob: Callable          # (guide_params, image, latents) -> (B,) log q
        guide_point_estimates: Callable   # (guide_params, image) -> dict of posterior means/probs
        render: Callable                  # (latents-dict) -> rendered RGB/gray image batch (for viz)
        model_log_joint: Callable         # (images, latents-dict) -> (n,) log p_model(theta,x) (for SC)
        composition_to_rgb: Callable      # (composition) -> RGB for diagnostics
        predictive_sites: tuple           # sites to return from Predictive
        extras: dict = field(default_factory=dict)

    MODEL_VERSIONS: dict = {}

    def register_version(v):
        MODEL_VERSIONS[v.name] = v
        return v

    return MODEL_VERSIONS, ModelVersion, register_version


@app.cell
def _(Predictive, jax, jnp, optax, random):
    # ---------- version-agnostic simulation / NPE ----------
    def simulate_pairs(version, n, key, chunk=4000):
        outs, start = [], 0
        while start < n:
            m = min(chunk, n - start)
            outs.append(Predictive(version.model, num_samples=m,
                                   return_sites=version.predictive_sites)(random.fold_in(key, start)))
            start += m
        return {k: jnp.concatenate([o[k] for o in outs], 0) for k in outs[0]}

    def npe_loss(version, guide_params, batch):
        latents = {n: batch[n] for n in version.site_names}
        return -jnp.mean(version.guide_log_prob(guide_params, batch["obs"], latents))

    def make_npe_trainer(version, lr=5e-4):
        opt = optax.adam(lr)

        @jax.jit
        def step(gp, st, batch):
            loss, grads = jax.value_and_grad(lambda p: npe_loss(version, p, batch))(gp)
            updates, st = opt.update(grads, st, gp)
            return optax.apply_updates(gp, updates), st, loss

        return opt, step

    def eval_loss_chunked(version, guide_params, data, chunk=256):
        n = data["obs"].shape[0]
        tot = 0.0
        for i in range(0, n, chunk):
            b = {k: v[i:i + chunk] for k, v in data.items()}
            tot += float(npe_loss(version, guide_params, b)) * min(chunk, n - i)
        return tot / n

    def train_npe(version, train, val, steps=6000, batch=128, lr=5e-4, key=None, eval_every=500):
        """Returns (best_guide_params, history). Tracks best val loss; chunked val eval."""
        key = random.PRNGKey(0) if key is None else key
        opt, step = make_npe_trainer(version, lr)
        gp = version.init_guide_params(random.fold_in(key, 1))
        st = opt.init(gp)
        n = train["obs"].shape[0]
        best = (eval_loss_chunked(version, gp, val), gp, 0)
        hist = [(0, best[0])]
        for i in range(1, steps + 1):
            idx = random.choice(random.fold_in(key, 100 + i), n, (batch,), replace=False)
            gp, st, _ = step(gp, st, {k: v[idx] for k, v in train.items()})
            if i % eval_every == 0 or i == steps:
                vl = eval_loss_chunked(version, gp, val)
                hist.append((i, vl))
                if vl < best[0]:
                    best = (vl, gp, i)
        return best[1], hist, best

    return (
        eval_loss_chunked,
        make_npe_trainer,
        npe_loss,
        simulate_pairs,
        train_npe,
    )


@app.cell
def _(handlers, jax, jnp, optax, random):
    # ---------- version-agnostic self-consistency ----------
    def sc_components(version, ratio_norm=None):
        if ratio_norm is None:
            ratio_norm = version.image_shape[0] * version.image_shape[1] * version.channels

        def sample_proposals(gp, images, n_samples, key):
            def draw(k):
                tr = handlers.trace(handlers.seed(lambda im: version.guide(im, gp), k)).get_trace(images)
                return {n: jax.lax.stop_gradient(tr[n]["value"]) for n in version.site_names}
            per = [draw(k) for k in random.split(key, n_samples)]
            return {n: jnp.stack([p[n] for p in per], 0) for n in version.site_names}

        def _flatten(samples):
            S, B = samples[version.site_names[0]].shape[:2]
            return S, B, {n: v.reshape((S * B,) + v.shape[2:]) for n, v in samples.items()}

        def _bcast(images, S):
            return jnp.broadcast_to(images[None], (S,) + images.shape).reshape((S * images.shape[0],) + images.shape[1:])

        def _chunked(fn, n, chunk):
            outs, i = [], 0
            while i < n:
                j = min(i + chunk, n)
                outs.append(fn(i, j))
                i = j
            return jnp.concatenate(outs, 0)

        def log_joint_samples(images, samples, chunk=64):
            S, B, flat = _flatten(samples)
            imgs = _bcast(images, S)
            return _chunked(lambda i, j: version.model_log_joint(imgs[i:j], {k: v[i:j] for k, v in flat.items()}),
                            imgs.shape[0], chunk).reshape((S, B))

        def guide_log_prob_samples(gp, images, samples, chunk=64):
            S, B, flat = _flatten(samples)
            imgs = _bcast(images, S)
            return _chunked(lambda i, j: version.guide_log_prob(gp, imgs[i:j], {k: v[i:j] for k, v in flat.items()}),
                            imgs.shape[0], chunk).reshape((S, B))

        def sc_loss(gp, images, frozen_samples, frozen_log_joint):
            log_q = guide_log_prob_samples(gp, images, frozen_samples)
            ratio = (frozen_log_joint - log_q) / ratio_norm
            return jnp.mean(jnp.var(ratio, axis=0))

        return dict(sample_proposals=sample_proposals, log_joint_samples=log_joint_samples,
                    guide_log_prob_samples=guide_log_prob_samples, sc_loss=sc_loss)

    def make_sc_trainer(sc, lr=2e-5, clip=10.0):
        opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))

        @jax.jit
        def step(gp, st, images, frozen_samples, frozen_log_joint):
            loss, grads = jax.value_and_grad(sc["sc_loss"])(gp, images, frozen_samples, frozen_log_joint)
            updates, st = opt.update(grads, st, gp)
            return optax.apply_updates(gp, updates), st, loss, optax.global_norm(grads)

        return opt, step

    return make_sc_trainer, sc_components


@app.cell
def _(np):
    # ---------- real-image loading: NATIVE crops, no resize, no colour preprocessing ----------
    def load_real_rgb_patches(path="example.jpg", patch=64, n_patches=6, seed=0, grayscale=False):
        from PIL import Image
        mode = "L" if grayscale else "RGB"
        full = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
        if grayscale:
            full = full[..., None]
        H, W = full.shape[:2]
        rng = np.random.default_rng(seed)
        patches, meta = [], []
        for _ in range(n_patches):
            y0 = int(rng.integers(0, H - patch)); x0 = int(rng.integers(0, W - patch))
            p = full[y0:y0 + patch, x0:x0 + patch]
            patches.append(p.astype(np.float32))
            meta.append({"y": y0, "x": x0})
        return np.stack(patches), meta

    return (load_real_rgb_patches,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Model generations (all four, faithful to notebook4)

        Each cell builds + registers one `ModelVersion` using the **exact** renderer/guide
        from notebook4 (renamed to generic site names: `obs`/`mean`/`count` + canonical
        latent names). Interpretable latents and a global/shared renderer throughout.

        - **v1** — toy grayscale 32², x-anchored 3 slots, isotropic Gaussian blobs, CNN guide.
        - **v2** — RGB 64², 16-slot grid, hand-composed fixed-palette coordinate-sprite
          renderer (8 local x/y basis fns). *Superseded* (the fixed palette is the rigidity
          v3/v4 remove) but kept faithfully. Uses a v3-style slot-aligned CNN guide.
        - **v3** — RGB 64², 16-slot grid, learned coordinate-MLP renderer with a global
          composition→RGB matrix, slot-aligned CNN guide.
        - **v4** — RGB 64², 8×8 free-placement grid, fully-learned random-Fourier coordinate
          renderer (no palette), slot-aligned CNN guide.
        """
    )
    return


@app.cell
def _(ModelVersion, dist, jax, jnn, jnp, numpyro, random, register_version):
    # ============================== v1 — toy grayscale 32x32, x-anchored slots, CNN guide ==============================
    def _build_v1():
        IMG = (32, 32); MAXO = 3; CDIM = 2
        INTENSITY = jnp.array([0.35, 1.15], dtype=jnp.float32)
        edges = jnp.linspace(0.0, 1.0, MAXO + 1, dtype=jnp.float32)
        PLOW = jnp.stack([jnp.zeros(MAXO), edges[:-1]], -1)
        PHIGH = jnp.stack([jnp.ones(MAXO), edges[1:]], -1)
        PSCALE = PHIGH - PLOW
        SLOW, SHIGH = 0.045, 0.16
        SITES = ("background", "observation_noise", "presence", "position", "size", "composition")
        HID = 160
        OUT = 2 + 2 + MAXO + MAXO * 2 * 2 + MAXO * 2 + MAXO * CDIM

        def grid(s=IMG):
            y = jnp.linspace(0, 1, s[0]); x = jnp.linspace(0, 1, s[1])
            yy, xx = jnp.meshgrid(y, x, indexing="ij"); return jnp.stack([yy, xx], -1)

        def comp_intensity(c): return jnp.sum(c * INTENSITY, -1)
        def comp_rgb(c):
            g = jnp.clip(comp_intensity(c) / 1.2, 0, 1); return jnp.stack([g, g, g], -1)

        def render(bg, presence, position, size, composition, s=IMG):
            d = grid(s)[None] - position[:, None, None, :]
            k = jnp.exp(-0.5 * jnp.sum(d ** 2, -1) / (size[:, None, None] ** 2 + 1e-6))
            inten = comp_intensity(composition)
            return bg + jnp.sum(presence[:, None, None] * inten[:, None, None] * k, 0)

        def render_from_estimates(e):
            return jax.vmap(lambda b, pr, po, si, c: render(b, pr, po, si, c))(
                e["background"], e["presence_probs"], e["position"], e["size"], e["composition"])

        def model(image=None):
            bg = numpyro.sample("background", dist.Uniform(0.0, 0.20))
            noise = numpyro.sample("observation_noise", dist.LogNormal(jnp.log(0.035), 0.25))
            presence = numpyro.sample("presence", dist.Bernoulli(0.55).expand([MAXO]).to_event(1))
            position = numpyro.sample("position", dist.Uniform(PLOW, PHIGH).to_event(2))
            size = numpyro.sample("size", dist.Uniform(SLOW * jnp.ones(MAXO), SHIGH * jnp.ones(MAXO)).to_event(1))
            composition = numpyro.sample("composition", dist.Dirichlet(2.0 * jnp.ones(CDIM)).expand([MAXO]).to_event(1))
            mean = render(bg, presence, position, size, composition)
            numpyro.deterministic("mean", mean); numpyro.deterministic("count", jnp.sum(presence))
            numpyro.sample("obs", dist.Normal(mean, noise).to_event(2), obs=image)

        def model_log_joint(images, lat):
            bg, noise = lat["background"], lat["observation_noise"]
            mean = jax.vmap(lambda b, pr, po, si, c: render(b, pr, po, si, c))(
                bg, lat["presence"], lat["position"], lat["size"], lat["composition"])
            lp = dist.Uniform(0.0, 0.20).log_prob(bg)
            lp += dist.LogNormal(jnp.log(0.035), 0.25).log_prob(noise)
            lp += dist.Bernoulli(0.55).expand([MAXO]).to_event(1).log_prob(lat["presence"])
            lp += dist.Uniform(PLOW, PHIGH).to_event(2).log_prob(lat["position"])
            lp += dist.Uniform(SLOW * jnp.ones(MAXO), SHIGH * jnp.ones(MAXO)).to_event(1).log_prob(lat["size"])
            lp += dist.Dirichlet(2.0 * jnp.ones(CDIM)).expand([MAXO]).to_event(1).log_prob(lat["composition"])
            lp += dist.Normal(mean, noise[:, None, None]).to_event(2).log_prob(images)
            return lp

        def spinv(y): return jnp.log(jnp.expm1(jnp.asarray(y, jnp.float32)))
        def posit(r, f=1e-3): return jnn.softplus(r) + f

        def init_guide_params(key, hidden=HID):
            kc1, kc2, kfc, kout = random.split(key, 4)
            bias = jnp.concatenate([
                jnp.array([spinv(3.0), spinv(3.0)]), jnp.array([jnp.log(0.035), spinv(0.35)]),
                jnp.full((MAXO,), jnp.log(0.55 / 0.45)), jnp.full((MAXO * 4,), spinv(2.0)),
                jnp.full((MAXO * 2,), spinv(2.0)), jnp.full((MAXO * CDIM,), spinv(2.0))]).astype(jnp.float32)
            return {"conv1": random.normal(kc1, (5, 5, 1, 16)) * jnp.sqrt(2.0 / 25), "bconv1": jnp.zeros(16),
                    "conv2": random.normal(kc2, (5, 5, 16, 32)) * jnp.sqrt(2.0 / (25 * 16)), "bconv2": jnp.zeros(32),
                    "w_fc": random.normal(kfc, (8 * 8 * 32, hidden)) * jnp.sqrt(2.0 / (8 * 8 * 32)), "b_fc": jnp.zeros(hidden),
                    "w_out": random.normal(kout, (hidden, OUT)) * 0.02, "b_out": bias}

        def ensure(im):
            im = jnp.asarray(im, jnp.float32); return (im[None], True) if im.ndim == 2 else (im, False)
        def conv(x, w, b, s=1):
            return jax.lax.conv_general_dilated(x, w, (s, s), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + b

        def raw(gp, image):
            ib, _ = ensure(image); f = ib[..., None]
            f = jnn.relu(conv(f, gp["conv1"], gp["bconv1"], 2))
            f = jnn.relu(conv(f, gp["conv2"], gp["bconv2"], 2))
            flat = f.reshape(f.shape[0], -1)
            h = jnn.relu(flat @ gp["w_fc"] + gp["b_fc"]); return h @ gp["w_out"] + gp["b_out"]

        def parse(gp, image):
            o = raw(gp, image); i = 0
            bg = o[:, i:i + 2]; i += 2; nz = o[:, i:i + 2]; i += 2
            pl = o[:, i:i + MAXO]; i += MAXO
            pr = o[:, i:i + MAXO * 4].reshape(-1, MAXO, 2, 2); i += MAXO * 4
            sr = o[:, i:i + MAXO * 2].reshape(-1, MAXO, 2); i += MAXO * 2
            cr = o[:, i:i + MAXO * CDIM].reshape(-1, MAXO, CDIM)
            return dict(ba=posit(bg[:, 0]), bb=posit(bg[:, 1]), nl=nz[:, 0], ns=posit(nz[:, 1], 0.02),
                        pl=pl, pa=posit(pr[..., 0]), pb=posit(pr[..., 1]),
                        sa=posit(sr[..., 0]), sb=posit(sr[..., 1]), cc=posit(cr))

        def dists(p):
            return {"background": dist.TransformedDistribution(dist.Beta(p["ba"], p["bb"]), dist.transforms.AffineTransform(0.0, 0.20)),
                    "observation_noise": dist.LogNormal(p["nl"], p["ns"]),
                    "presence": dist.Bernoulli(logits=p["pl"]).to_event(1),
                    "position": dist.TransformedDistribution(dist.Beta(p["pa"], p["pb"]), dist.transforms.AffineTransform(PLOW, PSCALE)).to_event(2),
                    "size": dist.TransformedDistribution(dist.Beta(p["sa"], p["sb"]), dist.transforms.AffineTransform(SLOW, SHIGH - SLOW)).to_event(1),
                    "composition": dist.Dirichlet(p["cc"]).to_event(1)}

        def guide(image, gp):
            ib, _ = ensure(image); d = dists(parse(gp, ib))
            with numpyro.plate("b", ib.shape[0]):
                for n in SITES: numpyro.sample(n, d[n])

        def guide_log_prob(gp, image, lat):
            ib, single = ensure(image); d = dists(parse(gp, ib)); terms = []
            for n in SITES:
                v = jnp.asarray(lat[n])
                if single:
                    if n in ("background", "observation_noise") and v.ndim == 0: v = v[None]
                    elif n in ("presence", "size") and v.ndim == 1: v = v[None, :]
                    elif n in ("position", "composition") and v.ndim == 2: v = v[None, :, :]
                terms.append(d[n].log_prob(v))
            t = sum(terms); return t[0] if single else t

        def guide_point_estimates(gp, image):
            p = parse(gp, image)
            return {"background": 0.20 * p["ba"] / (p["ba"] + p["bb"]),
                    "observation_noise": jnp.exp(p["nl"] + 0.5 * p["ns"] ** 2),
                    "presence_probs": jnn.sigmoid(p["pl"]),
                    "position": PLOW + PSCALE * p["pa"] / (p["pa"] + p["pb"]),
                    "size": SLOW + (SHIGH - SLOW) * p["sa"] / (p["sa"] + p["sb"]),
                    "composition": p["cc"] / jnp.sum(p["cc"], -1, keepdims=True)}

        return ModelVersion(
            name="v1", description="toy grayscale 32x32, x-anchored 3 slots, Gaussian-blob renderer, CNN guide",
            image_shape=IMG, channels=1, max_objects=MAXO, composition_dim=CDIM, site_names=SITES,
            position_low=PLOW, position_high=PHIGH, size_low=SLOW, size_high=SHIGH,
            model=model, init_guide_params=init_guide_params, guide=guide, guide_log_prob=guide_log_prob,
            guide_point_estimates=guide_point_estimates, render=render_from_estimates,
            model_log_joint=model_log_joint, composition_to_rgb=comp_rgb,
            predictive_sites=(*SITES, "mean", "obs", "count"))

    register_version(_build_v1())
    return


@app.cell
def _(ModelVersion, dist, jax, jnn, jnp, numpyro, random, register_version):
    # ============================== v2 — RGB 64x64, 16-slot grid, hand-composed fixed-palette sprite renderer ==============================
    # Faithful to notebook4 v2 (the superseded fixed-palette renderer). v2 had no trained
    # guide originally; we attach a v3-style slot-aligned CNN guide so it is runnable.
    def _build_v2():
        IMG = (64, 64); CH = 3; ROWS = COLS = 4; MAXO = ROWS * COLS; CDIM = 4
        SLOW, SHIGH = 0.025, 0.078
        ye = jnp.linspace(0, 1, ROWS + 1); xe = jnp.linspace(0, 1, COLS + 1)
        PLOW = jnp.array([[ye[r], xe[c]] for r in range(ROWS) for c in range(COLS)], jnp.float32)
        PHIGH = jnp.array([[ye[r + 1], xe[c + 1]] for r in range(ROWS) for c in range(COLS)], jnp.float32)
        PSCALE = PHIGH - PLOW; SB = 16 // ROWS
        BG_LO = jnp.array([0.56, 0.58, 0.42]); BG_HI = jnp.array([0.86, 0.86, 0.72])
        SITES = ("background_rgb", "observation_noise", "presence", "position", "size", "composition")

        MATERIAL = jnp.array([[0.95, 0.20, 0.30], [0.12, 0.35, 0.95], [0.95, 0.82, 0.18], [0.10, 0.72, 0.55]], jnp.float32)
        RIM = jnp.array([[0.70, 0.08, 0.72], [0.05, 0.18, 0.88], [0.95, 0.86, 0.20], [0.08, 0.80, 0.70]], jnp.float32)
        SHADOW = jnp.array([[0.42, 0.10, 0.62], [0.08, 0.16, 0.65], [0.45, 0.35, 0.08], [0.05, 0.34, 0.42]], jnp.float32)
        HIGHLIGHT = jnp.array([1.0, 0.96, 0.55], jnp.float32)

        def init_renderer():
            centres = jnp.array([[0.0, 0.0], [-0.35, -0.25], [0.42, -0.10], [-0.42, 0.36],
                                 [0.18, 0.28], [0.02, -0.72], [0.58, 0.36], [-0.70, 0.06]], jnp.float32)
            scales = jnp.array([[0.80, 0.78], [0.38, 0.48], [0.42, 0.72], [0.22, 0.25],
                                [0.36, 0.34], [0.34, 0.28], [0.26, 0.34], [0.22, 0.22]], jnp.float32)
            rgb_bias = jnp.array([[-0.02, -0.02, -0.02], [-0.16, -0.12, -0.24], [-0.12, -0.15, -0.22],
                                  [0.22, 0.20, 0.08], [0.02, 0.0, -0.02], [-0.06, -0.07, -0.16],
                                  [0.04, 0.10, 0.02], [-0.06, -0.03, -0.09]], jnp.float32)
            comp_rgb = jnp.stack([0.65 * (MATERIAL - 0.55), -0.18 * SHADOW, -0.15 * SHADOW,
                                  0.10 * (MATERIAL - 0.35) + 0.08 * HIGHLIGHT, 0.38 * (MATERIAL - 0.45),
                                  0.34 * (RIM - 0.45), 0.20 * (MATERIAL - 0.30) + 0.05 * HIGHLIGHT,
                                  -0.10 * SHADOW], 0).astype(jnp.float32)
            return {"basis_centres_yx": centres, "basis_scales_yx": scales, "basis_rgb_bias": rgb_bias,
                    "basis_comp_rgb": comp_rgb, "size_rgb_weights": jnp.array([0.05, 0.03, -0.04], jnp.float32)}

        RP = init_renderer()

        def grid(s=IMG):
            y = jnp.linspace(0, 1, s[0]); x = jnp.linspace(0, 1, s[1])
            yy, xx = jnp.meshgrid(y, x, indexing="ij"); return jnp.stack([yy, xx], -1)

        def comp_rgb(c): return jnp.clip(c @ MATERIAL, 0, 1)

        def basis_fn(local_yx, rp):
            delta = local_yx[..., None, :] - rp["basis_centres_yx"][None, None, None, :, :]
            return jnp.exp(-0.5 * jnp.sum((delta / rp["basis_scales_yx"][None, None, None, :, :]) ** 2, -1))

        def sprite_effect(position, size, comp, rp, s=IMG):
            local = (grid(s)[None] - position[:, None, None, :]) / (size[:, None, None, None] + 1e-6)
            basis = basis_fn(local, rp)
            basis_rgb = jnp.einsum("oc,kcr->okr", comp, rp["basis_comp_rgb"]) + rp["basis_rgb_bias"][None, :, :]
            size_eff = (size[:, None, None, None] - 0.05) * rp["size_rgb_weights"][None, None, None, :]
            return jnp.einsum("ohwk,okr->ohwr", basis, basis_rgb) + jnp.sum(basis, -1)[..., None] * size_eff

        def render(bg_rgb, presence, position, size, comp, rp=RP, s=IMG):
            bg = jnp.broadcast_to(bg_rgb, s + (CH,))
            eff = sprite_effect(position, size, comp, rp, s)
            return jnp.clip(bg + jnp.sum(presence[:, None, None, None] * eff, 0), 0, 1)

        def render_from_estimates(e):
            return jax.vmap(lambda b, pr, po, si, c: render(b, pr, po, si, c))(
                e["background_rgb"], e["presence_probs"], e["position"], e["size"], e["composition"])

        def model(image=None):
            bg = numpyro.sample("background_rgb", dist.Uniform(BG_LO, BG_HI).to_event(1))
            noise = numpyro.sample("observation_noise", dist.LogNormal(jnp.log(0.025), 0.30))
            presence = numpyro.sample("presence", dist.Bernoulli(0.38).expand([MAXO]).to_event(1))
            position = numpyro.sample("position", dist.Uniform(PLOW, PHIGH).to_event(2))
            size = numpyro.sample("size", dist.Uniform(SLOW * jnp.ones(MAXO), SHIGH * jnp.ones(MAXO)).to_event(1))
            comp = numpyro.sample("composition", dist.Dirichlet(1.2 * jnp.ones(CDIM)).expand([MAXO]).to_event(1))
            mean = render(bg, presence, position, size, comp)
            numpyro.deterministic("mean", mean); numpyro.deterministic("count", jnp.sum(presence))
            numpyro.sample("obs", dist.Normal(mean, noise).to_event(3), obs=image)

        def model_log_joint(images, lat):
            bg, noise = lat["background_rgb"], lat["observation_noise"]
            mean = jax.vmap(lambda b, pr, po, si, c: render(b, pr, po, si, c))(
                bg, lat["presence"], lat["position"], lat["size"], lat["composition"])
            lp = dist.Uniform(BG_LO, BG_HI).to_event(1).log_prob(bg)
            lp += dist.LogNormal(jnp.log(0.025), 0.30).log_prob(noise)
            lp += dist.Bernoulli(0.38).expand([MAXO]).to_event(1).log_prob(lat["presence"])
            lp += dist.Uniform(PLOW, PHIGH).to_event(2).log_prob(lat["position"])
            lp += dist.Uniform(SLOW * jnp.ones(MAXO), SHIGH * jnp.ones(MAXO)).to_event(1).log_prob(lat["size"])
            lp += dist.Dirichlet(1.2 * jnp.ones(CDIM)).expand([MAXO]).to_event(1).log_prob(lat["composition"])
            lp += dist.Normal(mean, noise[:, None, None, None]).to_event(3).log_prob(images)
            return lp

        # v3-style slot-aligned CNN guide (CDIM=4)
        HID = 160; SPO = 1 + 4 + 2 + CDIM; GOUT = 2 * CH + 2
        def spinv(y): return jnp.log(jnp.expm1(jnp.asarray(y, jnp.float32)))
        def posit(r, f=1e-3): return jnn.softplus(r) + f

        def init_guide_params(key, sh=HID):
            ks = random.split(key, 8)
            gb = jnp.concatenate([jnp.full((2 * CH,), spinv(4.0)), jnp.array([jnp.log(0.025), spinv(0.30)])]).astype(jnp.float32)
            pb = jnp.concatenate([jnp.array([jnp.log(0.38 / 0.62)]), jnp.full((4,), spinv(2.0)),
                                  jnp.full((2,), spinv(2.0)), jnp.full((CDIM,), spinv(1.2))]).astype(jnp.float32)
            return {"conv1": random.normal(ks[0], (5, 5, CH, 32)) * jnp.sqrt(2 / (25 * CH)), "bconv1": jnp.zeros(32),
                    "conv2": random.normal(ks[1], (3, 3, 32, 64)) * jnp.sqrt(2 / (9 * 32)), "bconv2": jnp.zeros(64),
                    "conv3": random.normal(ks[2], (3, 3, 64, 96)) * jnp.sqrt(2 / (9 * 64)), "bconv3": jnp.zeros(96),
                    "sw1": random.normal(ks[3], (96 * 2, sh)) * jnp.sqrt(2 / (96 * 2)), "sb1": jnp.zeros(sh),
                    "sw2": random.normal(ks[4], (sh, sh)) * jnp.sqrt(2 / sh), "sb2": jnp.zeros(sh),
                    "sw3": random.normal(ks[5], (sh, SPO)) * 0.02, "sb3": pb,
                    "gw": random.normal(ks[6], (96, GOUT)) * 0.02, "gb": gb}

        def ensure(im):
            im = jnp.asarray(im, jnp.float32); return (im[None], True) if im.ndim == 3 else (im, False)
        def conv(x, w, b, s=1):
            return jax.lax.conv_general_dilated(x, w, (s, s), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + b

        def raw(gp, image):
            ib, _ = ensure(image)
            f = jnn.relu(conv(ib, gp["conv1"], gp["bconv1"], 2))
            f = jnn.relu(conv(f, gp["conv2"], gp["bconv2"], 2))
            f = jnn.relu(conv(f, gp["conv3"], gp["bconv3"], 1))
            b, fd = f.shape[0], f.shape[-1]
            ps = f.reshape(b, ROWS, SB, COLS, SB, fd).mean((2, 4)).reshape(b, MAXO, fd)
            g = f.mean((1, 2))
            si = jnp.concatenate([ps, jnp.broadcast_to(g[:, None, :], ps.shape)], -1)
            h = jnn.relu(jnp.einsum("bof,fh->boh", si, gp["sw1"]) + gp["sb1"])
            h = jnn.relu(jnp.einsum("boh,hk->bok", h, gp["sw2"]) + gp["sb2"])
            per = jnp.einsum("bok,kq->boq", h, gp["sw3"]) + gp["sb3"]
            return per, g @ gp["gw"] + gp["gb"]

        def parse(gp, image):
            per, g = raw(gp, image)
            pr = per[..., 1:5].reshape(-1, MAXO, 2, 2); sr = per[..., 5:7]; cr = per[..., 7:7 + CDIM]
            bg = g[:, :2 * CH].reshape(-1, CH, 2); nz = g[:, 2 * CH:2 * CH + 2]
            return dict(ba=posit(bg[..., 0]), bb=posit(bg[..., 1]), nl=nz[:, 0], ns=posit(nz[:, 1], 0.02),
                        pl=per[..., 0], pa=posit(pr[..., 0]), pb=posit(pr[..., 1]),
                        sa=posit(sr[..., 0]), sb=posit(sr[..., 1]), cc=posit(cr))

        def dists(p):
            return {"background_rgb": dist.TransformedDistribution(dist.Beta(p["ba"], p["bb"]), dist.transforms.AffineTransform(BG_LO, BG_HI - BG_LO)).to_event(1),
                    "observation_noise": dist.LogNormal(p["nl"], p["ns"]),
                    "presence": dist.Bernoulli(logits=p["pl"]).to_event(1),
                    "position": dist.TransformedDistribution(dist.Beta(p["pa"], p["pb"]), dist.transforms.AffineTransform(PLOW, PSCALE)).to_event(2),
                    "size": dist.TransformedDistribution(dist.Beta(p["sa"], p["sb"]), dist.transforms.AffineTransform(SLOW, SHIGH - SLOW)).to_event(1),
                    "composition": dist.Dirichlet(p["cc"]).to_event(1)}

        def guide(image, gp):
            ib, _ = ensure(image); d = dists(parse(gp, ib))
            with numpyro.plate("b", ib.shape[0]):
                for n in SITES: numpyro.sample(n, d[n])

        def guide_log_prob(gp, image, lat):
            ib, single = ensure(image); d = dists(parse(gp, ib)); terms = []
            for n in SITES:
                v = jnp.asarray(lat[n])
                if single:
                    if n == "observation_noise" and v.ndim == 0: v = v[None]
                    elif n == "background_rgb" and v.ndim == 1: v = v[None, :]
                    elif n in ("presence", "size") and v.ndim == 1: v = v[None, :]
                    elif n in ("position", "composition") and v.ndim == 2: v = v[None, :, :]
                terms.append(d[n].log_prob(v))
            t = sum(terms); return t[0] if single else t

        def guide_point_estimates(gp, image):
            p = parse(gp, image)
            return {"background_rgb": BG_LO + (BG_HI - BG_LO) * p["ba"] / (p["ba"] + p["bb"]),
                    "observation_noise": jnp.exp(p["nl"] + 0.5 * p["ns"] ** 2),
                    "presence_probs": jnn.sigmoid(p["pl"]),
                    "position": PLOW + PSCALE * p["pa"] / (p["pa"] + p["pb"]),
                    "size": SLOW + (SHIGH - SLOW) * p["sa"] / (p["sa"] + p["sb"]),
                    "composition": p["cc"] / jnp.sum(p["cc"], -1, keepdims=True)}

        return ModelVersion(
            name="v2", description="RGB 64x64, 16-slot grid, hand-composed fixed-palette sprite renderer (superseded), slot-aligned CNN guide",
            image_shape=IMG, channels=CH, max_objects=MAXO, composition_dim=CDIM, site_names=SITES,
            position_low=PLOW, position_high=PHIGH, size_low=SLOW, size_high=SHIGH,
            model=model, init_guide_params=init_guide_params, guide=guide, guide_log_prob=guide_log_prob,
            guide_point_estimates=guide_point_estimates, render=render_from_estimates,
            model_log_joint=model_log_joint, composition_to_rgb=comp_rgb,
            predictive_sites=(*SITES, "mean", "obs", "count"))

    register_version(_build_v2())
    return


@app.cell
def _(ModelVersion, dist, jax, jnn, jnp, numpyro, random, register_version):
    # ============================== v3 — RGB 64x64, 16-slot grid, coord-MLP (comp->RGB matrix) ==============================
    def _build_v3():
        IMG = (64, 64); CH = 3; ROWS = COLS = 4; MAXO = ROWS * COLS; CDIM = 3
        SLOW, SHIGH = 0.025, 0.085
        ye = jnp.linspace(0, 1, ROWS + 1); xe = jnp.linspace(0, 1, COLS + 1)
        PLOW = jnp.array([[ye[r], xe[c]] for r in range(ROWS) for c in range(COLS)], jnp.float32)
        PHIGH = jnp.array([[ye[r + 1], xe[c + 1]] for r in range(ROWS) for c in range(COLS)], jnp.float32)
        PSCALE = PHIGH - PLOW
        BG_LO = jnp.array([0.45, 0.45, 0.35]); BG_HI = jnp.array([0.88, 0.88, 0.82])
        SITES = ("background_rgb", "observation_noise", "presence", "position", "size", "composition")

        def init_renderer(key, hidden=32, comp_scale=1.4):
            k1, k2, kc = random.split(key, 3)
            idim = 2 + 1 + CDIM + CH; odim = 1 + CH
            w1 = random.normal(k1, (idim, hidden)) / jnp.sqrt(idim)
            w2 = random.normal(k2, (hidden, odim)) / jnp.sqrt(hidden)
            cc = random.normal(kc, (CDIM, CH)); cc = comp_scale * cc / (jnp.std(cc) + 1e-6)
            w2 = w2.at[:, 1:].multiply(0.35)
            return {"w1": w1, "b1": jnp.zeros(hidden), "w2": w2,
                    "b2": jnp.array([-0.65, 0.0, 0.0, 0.0]), "composition_colour": cc,
                    "delta_scale": jnp.array(0.65), "locality_sigma": jnp.array(1.25)}

        RP = init_renderer(random.PRNGKey(3001))

        def grid(s=IMG):
            y = jnp.linspace(0, 1, s[0]); x = jnp.linspace(0, 1, s[1])
            yy, xx = jnp.meshgrid(y, x, indexing="ij"); return jnp.stack([yy, xx], -1)

        def comp_rgb(c): return jnn.sigmoid(c @ RP["composition_colour"])

        def coord_render(local_yx, size, comp, local_bg, rp=RP):
            no, h, w, _ = local_yx.shape
            sf = jnp.broadcast_to(jnp.log(size[:, None, None, None] / 0.05), (no, h, w, 1))
            cf = jnp.broadcast_to(comp[:, None, None, :], (no, h, w, CDIM))
            feats = jnp.concatenate([local_yx, sf, cf, local_bg], -1)
            hid = jnp.tanh(jnp.einsum("ohwf,fg->ohwg", feats, rp["w1"]) + rp["b1"])
            raw = jnp.einsum("ohwg,gc->ohwc", hid, rp["w2"]) + rp["b2"]
            ccd = jnp.tanh(comp @ rp["composition_colour"])
            rawd = raw[..., 1:] + ccd[:, None, None, :]
            loc = jnp.exp(-0.5 * jnp.sum((local_yx / rp["locality_sigma"]) ** 2, -1))
            alpha = loc[..., None] * jnn.sigmoid(raw[..., :1])
            return alpha, rp["delta_scale"] * jnp.tanh(rawd)

        def render(bg_rgb, presence, position, size, comp, rp=RP, s=IMG):
            bg = jnp.broadcast_to(bg_rgb, s + (CH,))
            local = (grid(s)[None] - position[:, None, None, :]) / (size[:, None, None, None] + 1e-6)
            lbg = jnp.broadcast_to(bg[None], (MAXO,) + s + (CH,))
            a, d = coord_render(local, size, comp, lbg, rp)
            return jnp.clip(bg + jnp.sum(presence[:, None, None, None] * a * d, 0), 0, 1)

        def render_from_estimates(e):
            return jax.vmap(lambda b, pr, po, si, c: render(b, pr, po, si, c))(
                e["background_rgb"], e["presence_probs"], e["position"], e["size"], e["composition"])

        def model(image=None):
            bg = numpyro.sample("background_rgb", dist.Uniform(BG_LO, BG_HI).to_event(1))
            noise = numpyro.sample("observation_noise", dist.LogNormal(jnp.log(0.018), 0.25))
            presence = numpyro.sample("presence", dist.Bernoulli(0.35).expand([MAXO]).to_event(1))
            position = numpyro.sample("position", dist.Uniform(PLOW, PHIGH).to_event(2))
            size = numpyro.sample("size", dist.Uniform(SLOW * jnp.ones(MAXO), SHIGH * jnp.ones(MAXO)).to_event(1))
            comp = numpyro.sample("composition", dist.Dirichlet(1.1 * jnp.ones(CDIM)).expand([MAXO]).to_event(1))
            mean = render(bg, presence, position, size, comp)
            numpyro.deterministic("mean", mean); numpyro.deterministic("count", jnp.sum(presence))
            numpyro.sample("obs", dist.Normal(mean, noise).to_event(3), obs=image)

        def model_log_joint(images, lat):
            bg, noise = lat["background_rgb"], lat["observation_noise"]
            mean = jax.vmap(lambda b, pr, po, si, c: render(b, pr, po, si, c))(
                bg, lat["presence"], lat["position"], lat["size"], lat["composition"])
            lp = dist.Uniform(BG_LO, BG_HI).to_event(1).log_prob(bg)
            lp += dist.LogNormal(jnp.log(0.018), 0.25).log_prob(noise)
            lp += dist.Bernoulli(0.35).expand([MAXO]).to_event(1).log_prob(lat["presence"])
            lp += dist.Uniform(PLOW, PHIGH).to_event(2).log_prob(lat["position"])
            lp += dist.Uniform(SLOW * jnp.ones(MAXO), SHIGH * jnp.ones(MAXO)).to_event(1).log_prob(lat["size"])
            lp += dist.Dirichlet(1.1 * jnp.ones(CDIM)).expand([MAXO]).to_event(1).log_prob(lat["composition"])
            lp += dist.Normal(mean, noise[:, None, None, None]).to_event(3).log_prob(images)
            return lp

        # ---- slot-aligned CNN guide (16x16 feature map -> per-slot MLP + global) ----
        HID = 160; SPO = 1 + 4 + 2 + CDIM; GOUT = 2 * CH + 2; SB = 16 // ROWS
        def spinv(y): return jnp.log(jnp.expm1(jnp.asarray(y, jnp.float32)))
        def posit(r, f=1e-3): return jnn.softplus(r) + f

        def init_guide_params(key, sh=HID):
            ks = random.split(key, 8)
            gb = jnp.concatenate([jnp.full((2 * CH,), spinv(4.0)), jnp.array([jnp.log(0.018), spinv(0.30)])]).astype(jnp.float32)
            pb = jnp.concatenate([jnp.array([jnp.log(0.35 / 0.65)]), jnp.full((4,), spinv(2.0)),
                                  jnp.full((2,), spinv(2.0)), jnp.full((CDIM,), spinv(1.2))]).astype(jnp.float32)
            return {"conv1": random.normal(ks[0], (5, 5, CH, 32)) * jnp.sqrt(2 / (25 * CH)), "bconv1": jnp.zeros(32),
                    "conv2": random.normal(ks[1], (3, 3, 32, 64)) * jnp.sqrt(2 / (9 * 32)), "bconv2": jnp.zeros(64),
                    "conv3": random.normal(ks[2], (3, 3, 64, 96)) * jnp.sqrt(2 / (9 * 64)), "bconv3": jnp.zeros(96),
                    "sw1": random.normal(ks[3], (96 * 2, sh)) * jnp.sqrt(2 / (96 * 2)), "sb1": jnp.zeros(sh),
                    "sw2": random.normal(ks[4], (sh, sh)) * jnp.sqrt(2 / sh), "sb2": jnp.zeros(sh),
                    "sw3": random.normal(ks[5], (sh, SPO)) * 0.02, "sb3": pb,
                    "gw": random.normal(ks[6], (96, GOUT)) * 0.02, "gb": gb}

        def ensure(im):
            im = jnp.asarray(im, jnp.float32); return (im[None], True) if im.ndim == 3 else (im, False)
        def conv(x, w, b, s=1):
            return jax.lax.conv_general_dilated(x, w, (s, s), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + b

        def raw(gp, image):
            ib, _ = ensure(image)
            f = jnn.relu(conv(ib, gp["conv1"], gp["bconv1"], 2))
            f = jnn.relu(conv(f, gp["conv2"], gp["bconv2"], 2))
            f = jnn.relu(conv(f, gp["conv3"], gp["bconv3"], 1))
            b, fd = f.shape[0], f.shape[-1]
            ps = f.reshape(b, ROWS, SB, COLS, SB, fd).mean((2, 4)).reshape(b, MAXO, fd)
            g = f.mean((1, 2))
            si = jnp.concatenate([ps, jnp.broadcast_to(g[:, None, :], ps.shape)], -1)
            h = jnn.relu(jnp.einsum("bof,fh->boh", si, gp["sw1"]) + gp["sb1"])
            h = jnn.relu(jnp.einsum("boh,hk->bok", h, gp["sw2"]) + gp["sb2"])
            per = jnp.einsum("bok,kq->boq", h, gp["sw3"]) + gp["sb3"]
            return per, g @ gp["gw"] + gp["gb"]

        def parse(gp, image):
            per, g = raw(gp, image)
            pr = per[..., 1:5].reshape(-1, MAXO, 2, 2); sr = per[..., 5:7]; cr = per[..., 7:7 + CDIM]
            bg = g[:, :2 * CH].reshape(-1, CH, 2); nz = g[:, 2 * CH:2 * CH + 2]
            return dict(ba=posit(bg[..., 0]), bb=posit(bg[..., 1]), nl=nz[:, 0], ns=posit(nz[:, 1], 0.02),
                        pl=per[..., 0], pa=posit(pr[..., 0]), pb=posit(pr[..., 1]),
                        sa=posit(sr[..., 0]), sb=posit(sr[..., 1]), cc=posit(cr))

        def dists(p):
            return {"background_rgb": dist.TransformedDistribution(dist.Beta(p["ba"], p["bb"]), dist.transforms.AffineTransform(BG_LO, BG_HI - BG_LO)).to_event(1),
                    "observation_noise": dist.LogNormal(p["nl"], p["ns"]),
                    "presence": dist.Bernoulli(logits=p["pl"]).to_event(1),
                    "position": dist.TransformedDistribution(dist.Beta(p["pa"], p["pb"]), dist.transforms.AffineTransform(PLOW, PSCALE)).to_event(2),
                    "size": dist.TransformedDistribution(dist.Beta(p["sa"], p["sb"]), dist.transforms.AffineTransform(SLOW, SHIGH - SLOW)).to_event(1),
                    "composition": dist.Dirichlet(p["cc"]).to_event(1)}

        def guide(image, gp):
            ib, _ = ensure(image); d = dists(parse(gp, ib))
            with numpyro.plate("b", ib.shape[0]):
                for n in SITES: numpyro.sample(n, d[n])

        def guide_log_prob(gp, image, lat):
            ib, single = ensure(image); d = dists(parse(gp, ib)); terms = []
            for n in SITES:
                v = jnp.asarray(lat[n])
                if single:
                    if n == "observation_noise" and v.ndim == 0: v = v[None]
                    elif n == "background_rgb" and v.ndim == 1: v = v[None, :]
                    elif n in ("presence", "size") and v.ndim == 1: v = v[None, :]
                    elif n in ("position", "composition") and v.ndim == 2: v = v[None, :, :]
                terms.append(d[n].log_prob(v))
            t = sum(terms); return t[0] if single else t

        def guide_point_estimates(gp, image):
            p = parse(gp, image)
            return {"background_rgb": BG_LO + (BG_HI - BG_LO) * p["ba"] / (p["ba"] + p["bb"]),
                    "observation_noise": jnp.exp(p["nl"] + 0.5 * p["ns"] ** 2),
                    "presence_probs": jnn.sigmoid(p["pl"]),
                    "position": PLOW + PSCALE * p["pa"] / (p["pa"] + p["pb"]),
                    "size": SLOW + (SHIGH - SLOW) * p["sa"] / (p["sa"] + p["sb"]),
                    "composition": p["cc"] / jnp.sum(p["cc"], -1, keepdims=True)}

        return ModelVersion(
            name="v3", description="RGB 64x64, 16-slot grid, coord-MLP renderer (comp->RGB matrix), slot-aligned CNN guide",
            image_shape=IMG, channels=CH, max_objects=MAXO, composition_dim=CDIM, site_names=SITES,
            position_low=PLOW, position_high=PHIGH, size_low=SLOW, size_high=SHIGH,
            model=model, init_guide_params=init_guide_params, guide=guide, guide_log_prob=guide_log_prob,
            guide_point_estimates=guide_point_estimates, render=render_from_estimates,
            model_log_joint=model_log_joint, composition_to_rgb=comp_rgb,
            predictive_sites=(*SITES, "mean", "obs", "count"))

    register_version(_build_v3())
    return


@app.cell
def _(ModelVersion, dist, jax, jnn, jnp, numpyro, random, register_version):
    # ============================== v4 — fully-learned renderer, 8x8 free-placement grid ==============================
    def _build_v4():
        IMG = (64, 64); CH = 3; GRID = 8; MAXO = GRID * GRID; CDIM = 3
        SLOW, SHIGH = 0.02, 0.40; PPROB = 7.0 / MAXO
        cy = (jnp.arange(GRID) + 0.0) / GRID
        gy, gx = jnp.meshgrid(cy, cy, indexing="ij")
        PLOW = jnp.stack([gy.ravel(), gx.ravel()], -1).astype(jnp.float32)
        PHIGH = PLOW + 1.0 / GRID; PSCALE = PHIGH - PLOW
        BG_LO = jnp.array([0.40, 0.40, 0.35]); BG_HI = jnp.array([0.80, 0.80, 0.75])
        SITES = ("background_rgb", "bg_gradient", "observation_noise", "presence", "position", "size", "composition")

        def init_renderer(key, hidden=64, n_fourier=6):
            ks = random.split(key, 6)
            ff = random.normal(ks[0], (2, n_fourier)) * 1.0
            idim = 2 * n_fourier + 1 + CDIM + CH
            w1 = random.normal(ks[1], (idim, hidden)) / jnp.sqrt(idim)
            cstart = 2 * n_fourier + 1
            w1 = w1.at[cstart:cstart + CDIM, :].multiply(7.0)
            return {"fourier_freqs": ff, "w1": w1, "b1": jnp.zeros(hidden),
                    "w2": random.normal(ks[2], (hidden, hidden)) / jnp.sqrt(hidden), "b2": jnp.zeros(hidden),
                    "w3": random.normal(ks[3], (hidden, 1 + CH)) * (3.5 / jnp.sqrt(hidden)),
                    "b3": jnp.concatenate([jnp.array([1.2]), jnp.zeros(CH)]), "locality_sigma": jnp.array(0.8)}

        RP = init_renderer(random.PRNGKey(4001))

        def grid_fn(s=IMG):
            y = jnp.linspace(0, 1, s[0]); x = jnp.linspace(0, 1, s[1])
            yy, xx = jnp.meshgrid(y, x, indexing="ij"); return jnp.stack([yy, xx], -1)

        def background_field(bg_rgb, grad, s=IMG):
            c = grid_fn(s) - 0.5
            return jnp.clip(bg_rgb[None, None, :] + jnp.einsum("hwk,kc->hwc", c, grad), 0, 1)

        def object_renderer(local_yx, size, comp, local_bg, rp=RP):
            no, h, w, _ = local_yx.shape
            proj = jnp.einsum("ohwk,kf->ohwf", local_yx, rp["fourier_freqs"])
            fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], -1)
            ls = jnp.broadcast_to(jnp.log(size[:, None, None, None] / 0.1), (no, h, w, 1))
            cf = jnp.broadcast_to(comp[:, None, None, :], (no, h, w, CDIM))
            feats = jnp.concatenate([fourier, ls, cf, local_bg], -1)
            z = jnp.tanh(jnp.einsum("ohwf,fg->ohwg", feats, rp["w1"]) + rp["b1"])
            z = jnp.tanh(jnp.einsum("ohwg,gk->ohwk", z, rp["w2"]) + rp["b2"])
            out = jnp.einsum("ohwk,kc->ohwc", z, rp["w3"]) + rp["b3"]
            loc = jnp.exp(-0.5 * jnp.sum((local_yx / rp["locality_sigma"]) ** 2, -1))
            return loc * jnn.sigmoid(out[..., 0]), jnn.sigmoid(out[..., 1:])

        def render(bg_rgb, grad, presence, position, size, comp, rp=RP, s=IMG):
            bg = background_field(bg_rgb, grad, s)
            local = (grid_fn(s)[None] - position[:, None, None, :]) / (size[:, None, None, None] + 1e-6)
            lbg = jnp.broadcast_to(bg[None], (position.shape[0],) + s + (CH,))
            a, rgb = object_renderer(local, size, comp, lbg, rp)
            ea = presence[:, None, None] * a
            ta = jnp.sum(ea, 0); wr = jnp.sum(ea[..., None] * rgb, 0)
            blend = wr / (ta[..., None] + 1e-6); cov = 1.0 - jnp.exp(-ta)
            return jnp.clip((1 - cov[..., None]) * bg + cov[..., None] * blend, 0, 1)

        def render_from_estimates(e):
            return jax.vmap(lambda b, g, pr, po, si, c: render(b, g, pr, po, si, c))(
                e["background_rgb"], e["bg_gradient"], e["presence_probs"], e["position"], e["size"], e["composition"])

        def comp_rgb(composition, size=0.15):
            flat = jnp.asarray(composition).reshape(-1, CDIM)
            def one(cv):
                pres = jnp.zeros(MAXO).at[27].set(1.0)
                comp = jnp.broadcast_to(jnp.ones(CDIM) / CDIM, (MAXO, CDIM)).at[27].set(cv)
                img = render(jnp.array([0.6, 0.6, 0.55]), jnp.zeros((2, 3)), pres,
                             0.5 * (PLOW + PHIGH), size * jnp.ones(MAXO), comp)
                c = 0.5 * (PLOW[27] + PHIGH[27]); py, px = (c * 63).astype(int)
                return jnp.mean(img[py - 2:py + 3, px - 2:px + 3], (0, 1))
            return jax.vmap(one)(flat).reshape(composition.shape[:-1] + (3,))

        def model(image=None):
            bg = numpyro.sample("background_rgb", dist.Uniform(BG_LO, BG_HI).to_event(1))
            grad = numpyro.sample("bg_gradient", dist.Normal(0.0, 0.08).expand([2, 3]).to_event(2))
            noise = numpyro.sample("observation_noise", dist.LogNormal(jnp.log(0.02), 0.3))
            presence = numpyro.sample("presence", dist.Bernoulli(PPROB).expand([MAXO]).to_event(1))
            position = numpyro.sample("position", dist.Uniform(PLOW, PHIGH).to_event(2))
            size = numpyro.sample("size", dist.TransformedDistribution(
                dist.Beta(1.3 * jnp.ones(MAXO), 3.0 * jnp.ones(MAXO)),
                dist.transforms.AffineTransform(SLOW, SHIGH - SLOW)).to_event(1))
            comp = numpyro.sample("composition", dist.Dirichlet(jnp.ones(CDIM)).expand([MAXO]).to_event(1))
            mean = render(bg, grad, presence, position, size, comp)
            numpyro.deterministic("mean", mean); numpyro.deterministic("count", jnp.sum(presence))
            numpyro.sample("obs", dist.Normal(mean, noise).to_event(3), obs=image)

        def model_log_joint(images, lat):
            bg, grad, noise = lat["background_rgb"], lat["bg_gradient"], lat["observation_noise"]
            mean = jax.vmap(lambda b, g, pr, po, si, c: render(b, g, pr, po, si, c))(
                bg, grad, lat["presence"], lat["position"], lat["size"], lat["composition"])
            lp = dist.Uniform(BG_LO, BG_HI).to_event(1).log_prob(bg)
            lp += dist.Normal(0.0, 0.08).expand([2, 3]).to_event(2).log_prob(grad)
            lp += dist.LogNormal(jnp.log(0.02), 0.3).log_prob(noise)
            lp += dist.Bernoulli(PPROB).expand([MAXO]).to_event(1).log_prob(lat["presence"])
            lp += dist.Uniform(PLOW, PHIGH).to_event(2).log_prob(lat["position"])
            lp += dist.TransformedDistribution(dist.Beta(1.3 * jnp.ones(MAXO), 3.0 * jnp.ones(MAXO)),
                                               dist.transforms.AffineTransform(SLOW, SHIGH - SLOW)).to_event(1).log_prob(lat["size"])
            lp += dist.Dirichlet(jnp.ones(CDIM)).expand([MAXO]).to_event(1).log_prob(lat["composition"])
            lp += dist.Normal(mean, noise[:, None, None, None]).to_event(3).log_prob(images)
            return lp

        # ---- slot-aligned CNN guide over 8x8 grid ----
        HID = 192; SPO = 1 + 4 + 2 + CDIM; GOUT = 2 * CH + 2 + 6
        def spinv(y): return jnp.log(jnp.expm1(jnp.asarray(y, jnp.float32)))
        def posit(r, f=1e-3): return jnn.softplus(r) + f

        def init_guide_params(key, sh=HID):
            ks = random.split(key, 9)
            gb = jnp.concatenate([jnp.full((2 * CH,), spinv(4.0)), jnp.array([jnp.log(0.02), spinv(0.30)]), jnp.zeros(6)]).astype(jnp.float32)
            pb = jnp.concatenate([jnp.array([jnp.log(PPROB / (1 - PPROB))]), jnp.full((4,), spinv(2.0)),
                                  jnp.full((2,), spinv(2.0)), jnp.full((CDIM,), spinv(1.0))]).astype(jnp.float32)
            return {"conv1": random.normal(ks[0], (5, 5, CH, 32)) * jnp.sqrt(2 / (25 * CH)), "bconv1": jnp.zeros(32),
                    "conv2": random.normal(ks[1], (3, 3, 32, 64)) * jnp.sqrt(2 / (9 * 32)), "bconv2": jnp.zeros(64),
                    "conv3": random.normal(ks[2], (3, 3, 64, 96)) * jnp.sqrt(2 / (9 * 64)), "bconv3": jnp.zeros(96),
                    "sw1": random.normal(ks[3], (96 * 2, sh)) * jnp.sqrt(2 / (96 * 2)), "sb1": jnp.zeros(sh),
                    "sw2": random.normal(ks[4], (sh, sh)) * jnp.sqrt(2 / sh), "sb2": jnp.zeros(sh),
                    "sw3": random.normal(ks[5], (sh, SPO)) * 0.02, "sb3": pb,
                    "gw": random.normal(ks[6], (96, GOUT)) * 0.02, "gb": gb}

        def ensure(im):
            im = jnp.asarray(im, jnp.float32); return (im[None], True) if im.ndim == 3 else (im, False)
        def conv(x, w, b, s=1):
            return jax.lax.conv_general_dilated(x, w, (s, s), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + b

        def raw(gp, image):
            ib, _ = ensure(image)
            f = jnn.relu(conv(ib, gp["conv1"], gp["bconv1"], 2))
            f = jnn.relu(conv(f, gp["conv2"], gp["bconv2"], 2))
            f = jnn.relu(conv(f, gp["conv3"], gp["bconv3"], 2))
            b, fd = f.shape[0], f.shape[-1]
            ps = f.reshape(b, GRID, GRID, fd).reshape(b, MAXO, fd)
            g = f.mean((1, 2))
            si = jnp.concatenate([ps, jnp.broadcast_to(g[:, None, :], ps.shape)], -1)
            h = jnn.relu(jnp.einsum("bof,fh->boh", si, gp["sw1"]) + gp["sb1"])
            h = jnn.relu(jnp.einsum("boh,hk->bok", h, gp["sw2"]) + gp["sb2"])
            per = jnp.einsum("bok,kq->boq", h, gp["sw3"]) + gp["sb3"]
            return per, g @ gp["gw"] + gp["gb"]

        def parse(gp, image):
            per, g = raw(gp, image)
            pr = per[..., 1:5].reshape(-1, MAXO, 2, 2); sr = per[..., 5:7]; cr = per[..., 7:7 + CDIM]
            bg = g[:, :2 * CH].reshape(-1, CH, 2); nz = g[:, 2 * CH:2 * CH + 2]; gr = g[:, 2 * CH + 2:].reshape(-1, 2, 3)
            return dict(ba=posit(bg[..., 0]), bb=posit(bg[..., 1]), nl=nz[:, 0], ns=posit(nz[:, 1], 0.02), gl=gr,
                        pl=per[..., 0], pa=posit(pr[..., 0]), pb=posit(pr[..., 1]),
                        sa=posit(sr[..., 0]), sb=posit(sr[..., 1]), cc=posit(cr))

        def dists(p):
            return {"background_rgb": dist.TransformedDistribution(dist.Beta(p["ba"], p["bb"]), dist.transforms.AffineTransform(BG_LO, BG_HI - BG_LO)).to_event(1),
                    "bg_gradient": dist.Normal(p["gl"], 0.05).to_event(2),
                    "observation_noise": dist.LogNormal(p["nl"], p["ns"]),
                    "presence": dist.Bernoulli(logits=p["pl"]).to_event(1),
                    "position": dist.TransformedDistribution(dist.Beta(p["pa"], p["pb"]), dist.transforms.AffineTransform(PLOW, PSCALE)).to_event(2),
                    "size": dist.TransformedDistribution(dist.Beta(p["sa"], p["sb"]), dist.transforms.AffineTransform(SLOW, SHIGH - SLOW)).to_event(1),
                    "composition": dist.Dirichlet(p["cc"]).to_event(1)}

        def guide(image, gp):
            ib, _ = ensure(image); d = dists(parse(gp, ib))
            with numpyro.plate("b", ib.shape[0]):
                for n in SITES: numpyro.sample(n, d[n])

        def guide_log_prob(gp, image, lat):
            ib, single = ensure(image); d = dists(parse(gp, ib)); terms = []
            for n in SITES:
                v = jnp.asarray(lat[n])
                if single:
                    if n == "observation_noise" and v.ndim == 0: v = v[None]
                    elif n == "background_rgb" and v.ndim == 1: v = v[None, :]
                    elif n == "bg_gradient" and v.ndim == 2: v = v[None, :, :]
                    elif n in ("presence", "size") and v.ndim == 1: v = v[None, :]
                    elif n in ("position", "composition") and v.ndim == 2: v = v[None, :, :]
                terms.append(d[n].log_prob(v))
            t = sum(terms); return t[0] if single else t

        def guide_point_estimates(gp, image):
            p = parse(gp, image)
            return {"background_rgb": BG_LO + (BG_HI - BG_LO) * p["ba"] / (p["ba"] + p["bb"]),
                    "bg_gradient": p["gl"],
                    "observation_noise": jnp.exp(p["nl"] + 0.5 * p["ns"] ** 2),
                    "presence_probs": jnn.sigmoid(p["pl"]),
                    "position": PLOW + PSCALE * p["pa"] / (p["pa"] + p["pb"]),
                    "size": SLOW + (SHIGH - SLOW) * p["sa"] / (p["sa"] + p["sb"]),
                    "composition": p["cc"] / jnp.sum(p["cc"], -1, keepdims=True)}

        return ModelVersion(
            name="v4", description="fully-learned random-Fourier coord renderer (no palette), 8x8 free-placement grid, slot-aligned CNN guide",
            image_shape=IMG, channels=CH, max_objects=MAXO, composition_dim=CDIM, site_names=SITES,
            position_low=PLOW, position_high=PHIGH, size_low=SLOW, size_high=SHIGH,
            model=model, init_guide_params=init_guide_params, guide=guide, guide_log_prob=guide_log_prob,
            guide_point_estimates=guide_point_estimates, render=render_from_estimates,
            model_log_joint=model_log_joint, composition_to_rgb=comp_rgb,
            predictive_sites=(*SITES, "mean", "obs", "count"))

    register_version(_build_v4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Version selector""")
    return


@app.cell
def _(MODEL_VERSIONS, mo):
    _order = ["v1", "v2", "v3", "v4"]
    _names = [n for n in _order if n in MODEL_VERSIONS] + [n for n in MODEL_VERSIONS if n not in _order]
    version_dropdown = mo.ui.dropdown(options=_names, value="v4" if "v4" in _names else _names[0],
                                      label="Active model generation")
    version_dropdown
    return (version_dropdown,)


@app.cell
def _(MODEL_VERSIONS, mo, version_dropdown):
    active_version = MODEL_VERSIONS[version_dropdown.value]
    mo.md(f"**Active:** `{active_version.name}` — {active_version.description}\n\n"
          f"image `{active_version.image_shape}`x`{active_version.channels}ch`, "
          f"`{active_version.max_objects}` max objects, composition dim `{active_version.composition_dim}`, "
          f"size `[{active_version.size_low}, {active_version.size_high}]`.")
    return (active_version,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8. Shared workflow (drives the active version)

        Helper to display a (gray or RGB) image batch uniformly.
        """
    )
    return


@app.cell
def _(np):
    def show_img(ax, img):
        a = np.asarray(img)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim == 2:
            ax.imshow(np.clip(a, 0, 1), cmap="magma", interpolation="nearest")
        else:
            ax.imshow(np.clip(a, 0, 1), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
    return (show_img,)


@app.cell
def _(Predictive, active_version, np, plt, random, show_img):
    # ---- 8a. Prior predictive grid ----
    _pp = Predictive(active_version.model, num_samples=16,
                     return_sites=("obs", "count"))(random.PRNGKey(1))
    _fig, _axes = plt.subplots(4, 4, figsize=(9, 9))
    for _ax, _im, _c in zip(_axes.ravel(), np.asarray(_pp["obs"]), np.asarray(_pp["count"])):
        show_img(_ax, _im); _ax.set_title(f"n={int(_c)}", fontsize=8)
    _fig.suptitle(f"{active_version.name} prior predictive", y=0.99)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 8b. NPE training (uncomment to run)

        Train the active version with the shared trainer, then check ground-truth latent
        recovery vs prior-mean baselines (gate on this, NOT image MSE). For v4 expect
        pos/size/colour to recover and count/presence to be the weak point.
        """
    )
    return


@app.cell
def _(active_version, jnp, random, simulate_pairs, train_npe):
    RUN_TRAINING = False  # set True to train the active version

    if RUN_TRAINING:
        _key = random.PRNGKey(0)
        train_data = simulate_pairs(active_version, 14000, random.fold_in(_key, 1))
        val_data = simulate_pairs(active_version, 2000, random.fold_in(_key, 2))
        trained_guide_params, train_history, train_best = train_npe(
            active_version, train_data, val_data, steps=6000, batch=128, lr=5e-4, key=_key)
        training_summary = {"init_val": train_history[0][1], "best_val": train_best[0], "best_step": train_best[2]}
    else:
        train_data = val_data = trained_guide_params = train_history = None
        training_summary = "RUN_TRAINING is False"
    training_summary
    return train_data, trained_guide_params, val_data


@app.cell
def _(active_version, jnp, np):
    # ---- 8c. Ground-truth latent recovery (slotwise) vs prior-mean baseline ----
    def latent_metrics(version, guide_params, data, n=1024):
        d = {k: v[:n] for k, v in data.items()}
        e = version.guide_point_estimates(guide_params, d["obs"])
        pres = d[[s for s in version.site_names if "presence" in s][0]]
        active = jnp.sum(pres)
        pos_t = d[[s for s in version.site_names if "position" in s][0]]
        size_t = d[[s for s in version.site_names if "size" in s][0]]
        comp_t = d[[s for s in version.site_names if "composition" in s][0]]
        pos_e, size_e, comp_e = e["position"], e["size"], e["composition"]
        pos = float(jnp.sum(jnp.sqrt(jnp.sum((pos_e - pos_t) ** 2, -1)) * pres) / (active + 1e-6))
        size = float(jnp.sum(jnp.abs(size_e - size_t) * pres) / (active + 1e-6))
        comp = float(jnp.sum(jnp.abs(comp_e - comp_t) * pres[:, :, None]) / (version.composition_dim * active + 1e-6))
        tcol = version.composition_to_rgb(comp_t); pcol = version.composition_to_rgb(comp_e)
        m = pres.astype(bool)
        colour_std_ratio = float(jnp.mean(jnp.std(pcol[m], 0) / (jnp.std(tcol[m], 0) + 1e-6)))
        hard = jnp.sum((e["presence_probs"] > 0.5).astype(jnp.float32), -1)
        count_t = d["count"]
        count_acc = float(jnp.mean(hard == count_t))
        count_mae = float(jnp.mean(jnp.abs(jnp.sum(e["presence_probs"], -1) - count_t)))
        centre = 0.5 * (np.asarray(version.position_low) + np.asarray(version.position_high))
        base_pos = float(jnp.sum(jnp.sqrt(jnp.sum((jnp.asarray(centre) - pos_t) ** 2, -1)) * pres) / (active + 1e-6))
        base_size = float(jnp.sum(jnp.abs(0.5 * (version.size_low + version.size_high) - size_t) * pres) / (active + 1e-6))
        return dict(pos=pos, size=size, comp=comp, colour_std_ratio=colour_std_ratio,
                    count_acc=count_acc, count_mae=count_mae, base_pos=base_pos, base_size=base_size)
    return (latent_metrics,)


@app.cell
def _(active_version, latent_metrics, mo, trained_guide_params, val_data):
    if trained_guide_params is not None:
        _val_m = latent_metrics(active_version, trained_guide_params, val_data)
        _out = mo.md(f"### {active_version.name} held-out latent recovery\n\n`{_val_m}`\n\n"
                     f"(pos must beat base_pos, size beat base_size, colour_std_ratio→1.)")
    else:
        _out = mo.md("_Train first (set `RUN_TRAINING=True`)._")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 8d. Real-image self-consistency (uncomment to run)

        Native 64×64 crops, no preprocessing (use grayscale=True for v1/v2). Tiny clipped
        SC steps; monitor a fixed-proposal SC value + synthetic non-collapse. `scp` the
        obs|fit|residual panel to inspect; compare v4 vs v3.
        """
    )
    return


@app.cell
def _(
    active_version,
    jnp,
    load_real_rgb_patches,
    make_sc_trainer,
    random,
    sc_components,
    trained_guide_params,
):
    RUN_SC = False  # set True after training to adapt to real patches

    if RUN_SC and trained_guide_params is not None:
        _gray = active_version.channels == 1
        _ps = active_version.image_shape[0]
        real_patches, _meta = load_real_rgb_patches(patch=_ps, n_patches=6, grayscale=_gray)
        real_images = jnp.asarray(real_patches if not _gray else real_patches[..., 0])
        sc = sc_components(active_version)
        opt_sc, step_sc = make_sc_trainer(sc, lr=2e-5, clip=10.0)
        gp_sc = trained_guide_params
        st_sc = opt_sc.init(gp_sc)
        _fixed = sc["sample_proposals"](gp_sc, real_images, 16, random.PRNGKey(50))
        _fixed_lj = sc["log_joint_samples"](real_images, _fixed)
        sc_before = float(sc["sc_loss"](gp_sc, real_images, _fixed, _fixed_lj))
        _k = random.PRNGKey(51)
        for _i in range(200):
            _k, _s = random.split(_k)
            _pr = sc["sample_proposals"](gp_sc, real_images, 8, _s)
            _lj = sc["log_joint_samples"](real_images, _pr)
            gp_sc, st_sc, _l, _gn = step_sc(gp_sc, st_sc, real_images, _pr, _lj)
        sc_after = float(sc["sc_loss"](gp_sc, real_images, _fixed, _fixed_lj))
        sc_summary = {"sc_before": sc_before, "sc_after": sc_after}
    else:
        real_images = gp_sc = None
        sc_summary = "RUN_SC is False (and needs a trained guide)"
    sc_summary
    return gp_sc, real_images


@app.cell
def _(active_version, gp_sc, np, plt, real_images, show_img):
    # ---- 8e. Real obs | SC fit | residual (after running SC) ----
    if real_images is not None and gp_sc is not None:
        _e = active_version.guide_point_estimates(gp_sc, real_images)
        _fit = active_version.render(_e)
        _n = real_images.shape[0]
        _fig, _axes = plt.subplots(_n, 3, figsize=(7, 2.4 * _n))
        for _r in range(_n):
            show_img(_axes[_r, 0], real_images[_r]); _axes[_r, 0].set_title("real" if _r == 0 else "", fontsize=9)
            show_img(_axes[_r, 1], _fit[_r]); _axes[_r, 1].set_title("SC fit" if _r == 0 else "", fontsize=9)
            _res = np.abs(np.asarray(real_images[_r]).reshape(_fit[_r].shape) - np.asarray(_fit[_r]))
            show_img(_axes[_r, 2], _res); _axes[_r, 2].set_title("|residual|" if _r == 0 else "", fontsize=9)
        _fig.suptitle(f"{active_version.name} self-consistency fit to real patches", y=0.999)
        _fig.tight_layout()
        _out = _fig
    else:
        import marimo as _mo
        _out = _mo.md("_Run SC first (set `RUN_SC=True`)._")
    _out
    return


if __name__ == "__main__":
    app.run()
