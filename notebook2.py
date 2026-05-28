# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "flax==0.12.7",
#     "jax[cuda12]==0.10.1",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "numpyro==0.21.0",
#     "optax==0.2.8",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import dataclasses
    from functools import partial

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive
    from matplotlib import pyplot as plt
    import dataclasses

    import flax.linen as linen
    from numpyro.contrib.module import flax_module as numpyro_flax_module
    import optax

    import os
    os.environ['JAX_PLATFORM'] = 'cpu'
    return (
        Predictive,
        dataclasses,
        dist,
        jax,
        jnp,
        jr,
        linen,
        np,
        numpyro,
        numpyro_flax_module,
        optax,
        plt,
    )


@app.cell
def _(Predictive, dataclasses, dist, jnp, jr, np, numpyro):
    # -------------------------------------------------------------------------
    # Minimal interpretable microscopy generative model
    #
    # Object semantics are intentionally hard-coded:
    #   object_i = (position_i, size_i, composition_i, presence_i)
    #
    # The renderer is order-invariant and low-capacity:
    #   image = background + sum_i spatial_kernel(position_i, size_i)
    #                        * colour_from_composition(composition_i)
    #
    # No learned neural renderer appears here. That is deliberate.
    # -------------------------------------------------------------------------

    @dataclasses.dataclass(frozen=True)
    class ModelConfig:
        tile_h: int = 96
        tile_w: int = 96
        max_objects: int = 32
        composition_dim: int = 3

        # Priors. Keep these as plain Python floats, not jax arrays.
        expected_objects: float = 12.0
        size_loc: float = float(np.log(6.0))
        size_scale: float = 0.35
        background_loc: float = 0.55
        background_scale: float = 0.20
        object_gain_loc: float = float(np.log(0.35))
        object_gain_scale: float = 0.35
        obs_sigma_loc: float = float(np.log(0.03))
        obs_sigma_scale: float = 0.25

    cfg = ModelConfig()

    def default_composition_basis(cfg: ModelConfig):
        """Anchor composition axes to RGB-like channels.

        Later, replace this with a calibrated basis or a strongly regularised
        global latent if the microscopy channels require it.
        """
        if cfg.composition_dim == 3:
            return jnp.eye(3, dtype=jnp.float32)
        return jnp.ones((cfg.composition_dim, 3), dtype=jnp.float32) / cfg.composition_dim

    COMPOSITION_BASIS = default_composition_basis(cfg)

    def pixel_grid(cfg: ModelConfig):
        ys = jnp.arange(cfg.tile_h, dtype=jnp.float32)
        xs = jnp.arange(cfg.tile_w, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        return jnp.stack([yy, xx], axis=-1)  # (H, W, 2)

    PIXELS = pixel_grid(cfg)

    def render_background(bg_corners, cfg: ModelConfig = cfg):
        """Bilinear RGB background from four corner colours.

        bg_corners shape: (4, 3), ordered as top-left, top-right,
        bottom-left, bottom-right.
        """
        h, w = cfg.tile_h, cfg.tile_w
        ys = jnp.linspace(0.0, 1.0, h)
        xs = jnp.linspace(0.0, 1.0, w)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

        tl, tr, bl, br = bg_corners
        top = (1.0 - xx)[..., None] * tl + xx[..., None] * tr
        bot = (1.0 - xx)[..., None] * bl + xx[..., None] * br
        return (1.0 - yy)[..., None] * top + yy[..., None] * bot

    def render_objects(
        position,
        size,
        composition,
        presence,
        composition_basis=COMPOSITION_BASIS,
        object_gain=1.0,
        cfg: ModelConfig = cfg,
    ):
        """Permutation-invariant object renderer.

        position:    (N, 2), y/x pixel coordinates
        size:        (N,), positive scalar Gaussian width in pixels
        composition: (N, K), simplex-valued composition vector
        presence:    (N,), binary or relaxed presence weight
        """
        delta = PIXELS[None, :, :, :] - position[:, None, None, :]  # (N, H, W, 2)
        r2 = jnp.sum(delta**2, axis=-1)                              # (N, H, W)

        size2 = jnp.maximum(size, 1e-3)[:, None, None] ** 2
        spatial = jnp.exp(-0.5 * r2 / size2)                         # (N, H, W)
        spatial = spatial * presence[:, None, None]

        object_rgb = composition @ composition_basis                  # (N, 3)
        contribution = spatial[..., None] * object_rgb[:, None, None, :]
        return object_gain * jnp.sum(contribution, axis=0)            # (H, W, 3)

    def render_latents(latents, cfg: ModelConfig = cfg):
        bg = render_background(latents["background"], cfg)
        obj = render_objects(
            position=latents["position"],
            size=latents["size"],
            composition=latents["composition"],
            presence=latents["presence"],
            composition_basis=latents["composition_basis"],
            object_gain=latents["object_gain"],
            cfg=cfg,
        )
        # Keep the first version simple. We can later replace this clipping with
        # a camera response model or an optical-density parameterisation.
        return jnp.clip(bg + obj, 0.0, 1.0)

    def microscopy_model(image=None, cfg: ModelConfig = cfg):
        """Interpretable NumPyro generative model for one image tile.

        Later stages will add an amortised guide and self-consistency training.
        This model already exposes the log joint needed for those stages.
        """
        max_objects = cfg.max_objects
        k = cfg.composition_dim

        # Strongly anchored for now. Do not make this a free neural renderer.
        composition_basis = numpyro.deterministic(
            "composition_basis",
            COMPOSITION_BASIS,
        )

        bg_logits = numpyro.sample(
            "background_logits",
            dist.Normal(cfg.background_loc, cfg.background_scale)
            .expand((4, 3))
            .to_event(2),
        )
        background = numpyro.deterministic("background", jnp.clip(bg_logits, 0.0, 1.0))

        object_gain = numpyro.sample(
            "object_gain",
            dist.LogNormal(cfg.object_gain_loc, cfg.object_gain_scale),
        )

        obs_sigma = numpyro.sample(
            "obs_sigma",
            dist.LogNormal(cfg.obs_sigma_loc, cfg.obs_sigma_scale),
        )

        # Bernoulli probability chosen so E[number present] ≈ expected_objects.
        p_present = jnp.clip(cfg.expected_objects / max_objects, 1e-3, 1.0 - 1e-3)

        with numpyro.plate("objects", max_objects):
            presence = numpyro.sample("presence", dist.Bernoulli(probs=p_present))

            position = numpyro.sample(
                "position",
                dist.Uniform(
                    low=jnp.array([0.0, 0.0]),
                    high=jnp.array([float(cfg.tile_h), float(cfg.tile_w)]),
                ).to_event(1),
            )

            size = numpyro.sample(
                "size",
                dist.LogNormal(cfg.size_loc, cfg.size_scale),
            )

            composition = numpyro.sample(
                "composition",
                dist.Dirichlet(jnp.ones(k)),
            )

        latents = {
            "background": background,
            "composition_basis": composition_basis,
            "object_gain": object_gain,
            "obs_sigma": obs_sigma,
            "presence": presence,
            "position": position,
            "size": size,
            "composition": composition,
        }

        mean = numpyro.deterministic("mean", render_latents(latents, cfg))

        numpyro.sample(
            "obs",
            dist.Normal(mean, obs_sigma).to_event(3),
            obs=image,
        )

        return latents

    def prior_predictive(key=jr.PRNGKey(0), num_samples=4):
        """Small smoke test for prior-predictive rendering."""
        return Predictive(
            microscopy_model,
            num_samples=num_samples,
            return_sites=[
                "obs",
                "mean",
                "presence",
                "position",
                "size",
                "composition",
                "background",
                "object_gain",
                "obs_sigma",
            ],
        )(key)

    print("Minimal interpretable NumPyro microscopy model defined.")
    print(f"tile={cfg.tile_h}x{cfg.tile_w}, max_objects={cfg.max_objects}, K={cfg.composition_dim}")
    return COMPOSITION_BASIS, cfg, microscopy_model, render_latents


@app.cell
def _(jnp, jr, np, plt, prior_predictive_localised):
    def show_prior_predictive(key=jr.PRNGKey(1), n=9, show_obs=False):
        samples = prior_predictive_localised(key, num_samples=n)
        site = "obs" if show_obs else "mean"

        imgs = np.asarray(jnp.clip(samples[site], 0.0, 1.0))
        presence = np.asarray(samples["presence"])
        sizes = np.asarray(samples["size"])
        object_gain = np.asarray(samples["object_gain"])
        obs_sigma = np.asarray(samples["obs_sigma"])

        counts = presence.sum(axis=1)
        present_sizes = sizes[presence.astype(bool)]

        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.asarray(axes).reshape(-1)

        for i, ax in enumerate(axes):
            ax.axis("off")
            if i >= n:
                continue
            ax.imshow(imgs[i])
            ax.set_title(
                f"{site} {i}: N={int(counts[i])}, "
                f"gain={object_gain[i]:.2f}, σ={obs_sigma[i]:.3f}"
            )

        fig.suptitle("Prior predictive samples", y=1.02)
        fig.tight_layout()

        print("Prior predictive summary")
        print(f"object count: mean={counts.mean():.2f}, min={counts.min():.0f}, max={counts.max():.0f}")
        print(f"object gain:  mean={object_gain.mean():.3f}, min={object_gain.min():.3f}, max={object_gain.max():.3f}")
        print(f"obs sigma:    mean={obs_sigma.mean():.4f}, min={obs_sigma.min():.4f}, max={obs_sigma.max():.4f}")

        if present_sizes.size:
            print(
                "size:         "
                f"mean={present_sizes.mean():.2f}, "
                f"p10={np.quantile(present_sizes, 0.10):.2f}, "
                f"p50={np.quantile(present_sizes, 0.50):.2f}, "
                f"p90={np.quantile(present_sizes, 0.90):.2f}"
            )

        return fig, samples

    fig_prior, prior_samples = show_prior_predictive(jr.PRNGKey(3), n=9, show_obs=False)

    fig_prior
    return


@app.cell
def _(Predictive, cfg, jax, jnp, jr, microscopy_model):
    from numpyro.infer.util import log_density


    LATENT_SAMPLE_SITES = [
        "background_logits",
        "object_gain",
        "obs_sigma",
        "presence",
        "position",
        "size",
        "composition",
    ]


    def sample_labelled_tiles(key, n=8):
        samples = Predictive(
            microscopy_model,
            num_samples=n,
            return_sites=LATENT_SAMPLE_SITES + ["obs", "mean"],
        )(key)

        latents = {name: samples[name] for name in LATENT_SAMPLE_SITES}
        images = samples["obs"]
        means = samples["mean"]
        return latents, images, means


    def slice_latents(latents, i):
        return {name: value[i] for name, value in latents.items()}


    def model_log_joint(theta, image):
        logp, _ = log_density(
            microscopy_model,
            model_args=(image,),
            model_kwargs={"cfg": cfg},
            params=theta,
        )
        return logp


    def perturb_latents(key, theta):
        k_pos, k_size, k_comp = jr.split(key, 3)

        out = dict(theta)
        out["position"] = jnp.clip(
            theta["position"] + 8.0 * jr.normal(k_pos, theta["position"].shape),
            jnp.array([0.0, 0.0]),
            jnp.array([float(cfg.tile_h), float(cfg.tile_w)]),
        )
        out["size"] = jnp.clip(
            theta["size"] * jnp.exp(0.35 * jr.normal(k_size, theta["size"].shape)),
            1.0,
            30.0,
        )

        comp_noise = 0.25 * jr.normal(k_comp, theta["composition"].shape)
        comp_logits = jnp.log(theta["composition"] + 1e-6) + comp_noise
        out["composition"] = jax.nn.softmax(comp_logits, axis=-1)

        return out


    def check_true_latent_score(key=jr.PRNGKey(11), n=8):
        latents, images, means = sample_labelled_tiles(key, n=n)
        keys = jr.split(jr.fold_in(key, 1), n)

        true_scores = []
        pert_scores = []

        for i in range(n):
            theta = slice_latents(latents, i)
            theta_bad = perturb_latents(keys[i], theta)

            true_scores.append(model_log_joint(theta, images[i]))
            pert_scores.append(model_log_joint(theta_bad, images[i]))

        true_scores = jnp.asarray(true_scores)
        pert_scores = jnp.asarray(pert_scores)

        print("log p(theta, x) sanity check")
        print(f"true:      mean={true_scores.mean():.2f}, min={true_scores.min():.2f}, max={true_scores.max():.2f}")
        print(f"perturbed: mean={pert_scores.mean():.2f}, min={pert_scores.min():.2f}, max={pert_scores.max():.2f}")
        print(f"true > perturbed: {int((true_scores > pert_scores).sum())}/{n}")

        return latents, images, means, true_scores, pert_scores


    synthetic_latents, synthetic_images, synthetic_means, true_scores, pert_scores = check_true_latent_score()
    return (
        LATENT_SAMPLE_SITES,
        log_density,
        pert_scores,
        perturb_latents,
        slice_latents,
        synthetic_images,
        synthetic_latents,
        synthetic_means,
        true_scores,
    )


@app.cell
def _(
    COMPOSITION_BASIS,
    jnp,
    jr,
    np,
    pert_scores,
    perturb_latents,
    plt,
    render_latents,
    slice_latents,
    synthetic_images,
    synthetic_latents,
    synthetic_means,
    true_scores,
):
    def theta_to_render_latents(theta):
        return {
            "background": jnp.clip(theta["background_logits"], 0.0, 1.0),
            "composition_basis": COMPOSITION_BASIS,
            "object_gain": theta["object_gain"],
            "obs_sigma": theta["obs_sigma"],
            "presence": theta["presence"],
            "position": theta["position"],
            "size": theta["size"],
            "composition": theta["composition"],
        }


    def draw_objects(ax, theta, colour="lime", alpha=0.8):
        presence = np.asarray(theta["presence"])
        position = np.asarray(theta["position"])
        size = np.asarray(theta["size"])

        for p, (y, x), s in zip(presence, position, size):
            if p < 0.5:
                continue
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    s,
                    fill=False,
                    edgecolor=colour,
                    linewidth=1.2,
                    alpha=alpha,
                )
            )


    def visualise_log_joint_check(
        latents=synthetic_latents,
        images=synthetic_images,
        means=synthetic_means,
        true_scores=true_scores,
        pert_scores=pert_scores,
        key=jr.PRNGKey(101),
        n=4,
    ):
        keys = jr.split(key, n)

        fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n), squeeze=False)

        for row in range(n):
            theta = slice_latents(latents, row)
            theta_bad = perturb_latents(keys[row], theta)

            true_img = np.asarray(jnp.clip(means[row], 0.0, 1.0))
            obs_img = np.asarray(jnp.clip(images[row], 0.0, 1.0))
            bad_img = np.asarray(jnp.clip(render_latents(theta_to_render_latents(theta_bad)), 0.0, 1.0))

            true_resid = np.mean(np.abs(obs_img - true_img), axis=-1)
            bad_resid = np.mean(np.abs(obs_img - bad_img), axis=-1)

            ax = axes[row, 0]
            ax.imshow(obs_img)
            ax.set_title("Simulated observation")
            ax.axis("off")

            ax = axes[row, 1]
            ax.imshow(true_img)
            draw_objects(ax, theta, colour="lime")
            ax.set_title(f"True latents\nlogp={float(true_scores[row]):.0f}")
            ax.axis("off")

            ax = axes[row, 2]
            ax.imshow(bad_img)
            draw_objects(ax, theta_bad, colour="magenta")
            ax.set_title(f"Perturbed latents\nlogp={float(pert_scores[row]):.0f}")
            ax.axis("off")

            ax = axes[row, 3]
            ax.imshow(true_resid, vmin=0.0, vmax=max(0.15, float(true_resid.max())))
            ax.set_title(f"|obs - true|\nmean={true_resid.mean():.4f}")
            ax.axis("off")

            ax = axes[row, 4]
            ax.imshow(bad_resid, vmin=0.0, vmax=max(0.15, float(bad_resid.max())))
            ax.set_title(f"|obs - perturbed|\nmean={bad_resid.mean():.4f}")
            ax.axis("off")

        fig.suptitle("Log-joint sanity check: true versus perturbed latents", y=1.01)
        fig.tight_layout()
        return fig


    fig_log_joint_check = visualise_log_joint_check(n=4)

    fig_log_joint_check
    return


@app.cell
def _(
    COMPOSITION_BASIS,
    LATENT_SAMPLE_SITES,
    Predictive,
    cfg,
    dist,
    jax,
    jnp,
    jr,
    np,
    numpyro,
    plt,
    render_latents,
):
    def _logit(x, eps=1e-5):
        x = jnp.clip(x, eps, 1.0 - eps)
        return jnp.log(x) - jnp.log1p(-x)


    def slot_anchors(cfg=cfg):
        n = cfg.max_objects
        rows = int(np.floor(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        ys = jnp.linspace(0.5 / rows, 1.0 - 0.5 / rows, rows)
        xs = jnp.linspace(0.5 / cols, 1.0 - 0.5 / cols, cols)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

        anchors01 = jnp.stack([yy.ravel(), xx.ravel()], axis=-1)[:n]
        scale = jnp.array([cfg.tile_h, cfg.tile_w], dtype=jnp.float32)
        return anchors01 * scale


    SLOT_ANCHORS = slot_anchors(cfg)
    SLOT_ANCHORS01 = SLOT_ANCHORS / jnp.array([cfg.tile_h, cfg.tile_w])
    SLOT_ANCHOR_LOGITS = _logit(SLOT_ANCHORS01)

    POSITION_OFFSET_SCALE = 0.85


    def microscopy_model_localised(image=None, cfg=cfg):
        max_objects = cfg.max_objects
        k = cfg.composition_dim

        composition_basis = numpyro.deterministic(
            "composition_basis",
            COMPOSITION_BASIS,
        )

        bg_logits = numpyro.sample(
            "background_logits",
            dist.Normal(cfg.background_loc, cfg.background_scale)
            .expand((4, 3))
            .to_event(2),
        )
        background = numpyro.deterministic("background", jnp.clip(bg_logits, 0.0, 1.0))

        object_gain = numpyro.sample(
            "object_gain",
            dist.LogNormal(cfg.object_gain_loc, cfg.object_gain_scale),
        )

        obs_sigma = numpyro.sample(
            "obs_sigma",
            dist.LogNormal(cfg.obs_sigma_loc, cfg.obs_sigma_scale),
        )

        p_present = jnp.clip(cfg.expected_objects / max_objects, 1e-3, 1.0 - 1e-3)

        with numpyro.plate("objects", max_objects):
            presence = numpyro.sample("presence", dist.Bernoulli(probs=p_present))

            position_offset = numpyro.sample(
                "position_offset",
                dist.Normal(0.0, POSITION_OFFSET_SCALE)
                .expand((2,))
                .to_event(1),
            )

            position01 = jax.nn.sigmoid(SLOT_ANCHOR_LOGITS + position_offset)
            position = numpyro.deterministic(
                "position",
                position01 * jnp.array([cfg.tile_h, cfg.tile_w]),
            )

            size = numpyro.sample(
                "size",
                dist.LogNormal(cfg.size_loc, cfg.size_scale),
            )

            composition = numpyro.sample(
                "composition",
                dist.Dirichlet(jnp.ones(k)),
            )

        latents = {
            "background": background,
            "composition_basis": composition_basis,
            "object_gain": object_gain,
            "obs_sigma": obs_sigma,
            "presence": presence,
            "position": position,
            "size": size,
            "composition": composition,
        }

        mean = numpyro.deterministic("mean", render_latents(latents, cfg))

        numpyro.sample(
            "obs",
            dist.Normal(mean, obs_sigma).to_event(3),
            obs=image,
        )

        return latents

    def prior_predictive_localised(key=jr.PRNGKey(0), num_samples=4):
        return Predictive(
            microscopy_model_localised,
            num_samples=num_samples,
            return_sites=LATENT_SAMPLE_SITES + [
                "obs",
                "mean",
                "position",
                "background",
                "composition_basis",
            ],
        )(key)


    def show_slot_anchors():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(jnp.zeros((cfg.tile_h, cfg.tile_w, 3)))
        ax.scatter(
            np.asarray(SLOT_ANCHORS[:, 1]),
            np.asarray(SLOT_ANCHORS[:, 0]),
            s=30,
        )
        for i, (y, x) in enumerate(np.asarray(SLOT_ANCHORS)):
            ax.text(x + 1, y + 1, str(i), fontsize=8)
        ax.set_title("Weak spatial anchors for object slots")
        ax.set_xlim(0, cfg.tile_w)
        ax.set_ylim(cfg.tile_h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig


    fig_slot_anchors = show_slot_anchors()
    fig_slot_anchors
    return microscopy_model_localised, prior_predictive_localised


@app.cell
def _(Predictive, jnp, jr, microscopy_model_localised, np, plt):
    latent_sample_sites_localised = [
        "background_logits",
        "object_gain",
        "obs_sigma",
        "presence",
        "position_offset",
        "size",
        "composition",
    ]

    display_sites_localised = latent_sample_sites_localised + [
        "obs",
        "mean",
        "position",
        "background",
        "composition_basis",
    ]


    def sample_labelled_tiles_localised(key, n=8):
        samples = Predictive(
            microscopy_model_localised,
            num_samples=n,
            return_sites=display_sites_localised,
        )(key)

        theta = {name: samples[name] for name in latent_sample_sites_localised}
        images = samples["obs"]
        means = samples["mean"]
        positions = samples["position"]

        return theta, images, means, positions, samples


    def slice_tree_localised(tree, i):
        return {name: value[i] for name, value in tree.items()}


    def draw_objects_localised(ax, position, presence, size, alpha=0.85):
        position_np = np.asarray(position)
        presence_np = np.asarray(presence)
        size_np = np.asarray(size)

        for p, (y, x), s in zip(presence_np, position_np, size_np):
            if p < 0.5:
                continue
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    s,
                    fill=False,
                    linewidth=1.2,
                    alpha=alpha,
                )
            )


    def visualise_labelled_tiles_localised(key=jr.PRNGKey(21), n=6):
        theta, images, means, positions, samples = sample_labelled_tiles_localised(key, n=n)

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

        for i in range(n):
            image = np.asarray(jnp.clip(images[i], 0.0, 1.0))
            mean = np.asarray(jnp.clip(means[i], 0.0, 1.0))
            resid = np.mean(np.abs(image - mean), axis=-1)

            presence_i = theta["presence"][i]
            size_i = theta["size"][i]
            position_i = positions[i]
            comp_i = theta["composition"][i]

            count_i = int(np.asarray(presence_i).sum())
            mean_size_i = float(np.asarray(size_i)[np.asarray(presence_i).astype(bool)].mean()) if count_i else 0.0
            mean_comp_i = np.asarray(
                (comp_i * presence_i[:, None]).sum(axis=0) / (presence_i.sum() + 1e-6)
            )

            ax = axes[i, 0]
            ax.imshow(image)
            ax.set_title(f"Observation {i}")
            ax.axis("off")

            ax = axes[i, 1]
            ax.imshow(mean)
            draw_objects_localised(ax, position_i, presence_i, size_i)
            ax.set_title(f"Mean + objects\nN={count_i}, size≈{mean_size_i:.1f}")
            ax.axis("off")

            ax = axes[i, 2]
            ax.imshow(resid)
            ax.set_title(
                "Residual + mean comp\n"
                f"[{mean_comp_i[0]:.2f}, {mean_comp_i[1]:.2f}, {mean_comp_i[2]:.2f}]"
            )
            ax.axis("off")

        fig.suptitle("Labelled synthetic tiles from localised model", y=1.01)
        fig.tight_layout()

        return fig, theta, images, means, positions, samples


    fig_labelled_localised, theta_labelled_localised, images_labelled_localised, means_labelled_localised, positions_labelled_localised, samples_labelled_localised = visualise_labelled_tiles_localised()

    fig_labelled_localised
    return (
        images_labelled_localised,
        sample_labelled_tiles_localised,
        slice_tree_localised,
        theta_labelled_localised,
    )


@app.cell
def _(
    cfg,
    dist,
    images_labelled_localised,
    jax,
    jnp,
    jr,
    linen,
    numpyro,
    numpyro_flax_module,
):
    GUIDE_EMB_DIM_LOCALISED = 128
    GUIDE_HIDDEN_LOCALISED = 128
    GUIDE_MIN_SCALE_LOCALISED = 1e-3
    GUIDE_MAX_SCALE_LOCALISED = 2.0
    GUIDE_MIN_CONC_LOCALISED = 1e-2


    def guide_positive_scale_localised(raw):
        return jnp.clip(
            GUIDE_MIN_SCALE_LOCALISED + jax.nn.softplus(raw),
            GUIDE_MIN_SCALE_LOCALISED,
            GUIDE_MAX_SCALE_LOCALISED,
        )


    class LocalisedEncoder(linen.Module):
        emb_dim: int = GUIDE_EMB_DIM_LOCALISED

        @linen.compact
        def __call__(self, image):
            x = image[None, ...]
            for channels in (16, 32, 64):
                x = linen.Conv(channels, (3, 3), strides=(2, 2), padding="SAME")(x)
                x = linen.gelu(x)
            x = x.mean(axis=(1, 2))[0]
            x = linen.Dense(self.emb_dim)(x)
            x = linen.gelu(x)
            return x


    class LocalisedPosteriorHead(linen.Module):
        max_objects: int
        composition_dim: int
        hidden: int = GUIDE_HIDDEN_LOCALISED

        @linen.compact
        def __call__(self, emb):
            h = linen.Dense(self.hidden)(emb)
            h = linen.gelu(h)
            h = linen.Dense(self.hidden)(h)
            h = linen.gelu(h)

            n = self.max_objects
            k = self.composition_dim

            background = linen.Dense(2 * 4 * 3)(h).reshape(4, 3, 2)
            object_gain = linen.Dense(2)(h)
            obs_sigma = linen.Dense(2)(h)

            presence_logits = linen.Dense(n)(h)

            position_offset = linen.Dense(2 * n * 2)(h).reshape(n, 2, 2)
            log_size = linen.Dense(2 * n)(h).reshape(n, 2)

            composition_raw = linen.Dense(n * k)(h).reshape(n, k)

            return {
                "background": background,
                "object_gain": object_gain,
                "obs_sigma": obs_sigma,
                "presence_logits": presence_logits,
                "position_offset": position_offset,
                "log_size": log_size,
                "composition_raw": composition_raw,
            }


    localised_encoder_module = LocalisedEncoder()
    localised_posterior_head_module = LocalisedPosteriorHead(
        max_objects=cfg.max_objects,
        composition_dim=cfg.composition_dim,
    )


    def localised_amortised_guide(image, cfg=cfg):
        encoder = numpyro_flax_module(
            "localised_encoder",
            localised_encoder_module,
            input_shape=(cfg.tile_h, cfg.tile_w, 3),
        )
        posterior_head = numpyro_flax_module(
            "localised_posterior_head",
            localised_posterior_head_module,
            input_shape=(GUIDE_EMB_DIM_LOCALISED,),
        )

        emb = encoder(image)
        q = posterior_head(emb)

        bg_loc = q["background"][..., 0]
        bg_scale = guide_positive_scale_localised(q["background"][..., 1])
        numpyro.sample(
            "background_logits",
            dist.Normal(bg_loc, bg_scale).to_event(2),
        )

        object_gain_loc = q["object_gain"][0]
        object_gain_scale = guide_positive_scale_localised(q["object_gain"][1])
        numpyro.sample(
            "object_gain",
            dist.LogNormal(object_gain_loc, object_gain_scale),
        )

        obs_sigma_loc = q["obs_sigma"][0]
        obs_sigma_scale = guide_positive_scale_localised(q["obs_sigma"][1])
        numpyro.sample(
            "obs_sigma",
            dist.LogNormal(obs_sigma_loc, obs_sigma_scale),
        )

        with numpyro.plate("objects", cfg.max_objects):
            numpyro.sample(
                "presence",
                dist.Bernoulli(logits=q["presence_logits"]),
            )

            position_offset_loc = q["position_offset"][..., 0]
            position_offset_scale = guide_positive_scale_localised(q["position_offset"][..., 1])
            numpyro.sample(
                "position_offset",
                dist.Normal(position_offset_loc, position_offset_scale).to_event(1),
            )

            log_size_loc = q["log_size"][..., 0]
            log_size_scale = guide_positive_scale_localised(q["log_size"][..., 1])
            numpyro.sample(
                "size",
                dist.LogNormal(log_size_loc, log_size_scale),
            )

            composition_concentration = (
                GUIDE_MIN_CONC_LOCALISED
                + jax.nn.softplus(q["composition_raw"])
            )
            numpyro.sample(
                "composition",
                dist.Dirichlet(composition_concentration),
            )


    def inspect_localised_guide_shapes(image=images_labelled_localised[0]):
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(localised_amortised_guide, jr.PRNGKey(0))
        ).get_trace(image)

        return {
            name: {
                "shape": tuple(site["value"].shape),
                "fn": type(site["fn"]).__name__,
            }
            for name, site in trace.items()
            if site["type"] == "sample"
        }


    guide_shape_summary_localised = inspect_localised_guide_shapes()
    guide_shape_summary_localised
    return (localised_amortised_guide,)


@app.cell
def _(
    cfg,
    images_labelled_localised,
    jnp,
    localised_amortised_guide,
    log_density,
    numpyro,
    slice_tree_localised,
    theta_labelled_localised,
):
    def init_localised_guide_params(image):
        seeded_guide = numpyro.handlers.seed(localised_amortised_guide, rng_seed=0)

        with numpyro.handlers.trace() as tr:
            seeded_guide(image)

        return {
            name: site["value"]
            for name, site in tr.items()
            if site["type"] == "param"
        }


    def localised_guide_log_prob(guide_params, theta, image):
        values = {**guide_params, **theta}

        logq, _ = log_density(
            localised_amortised_guide,
            model_args=(image,),
            model_kwargs={"cfg": cfg},
            params=values,
        )

        return logq


    def check_initial_localised_guide_density(
        guide_params,
        theta_batch=theta_labelled_localised,
        images=images_labelled_localised,
    ):
        logqs = []

        for i in range(images.shape[0]):
            theta_i = slice_tree_localised(theta_batch, i)
            logqs.append(localised_guide_log_prob(guide_params, theta_i, images[i]))

        logqs = jnp.asarray(logqs)

        print("Initial amortised guide density check")
        print(f"log q(theta | x): mean={logqs.mean():.2f}, min={logqs.min():.2f}, max={logqs.max():.2f}")
        print(f"all finite: {bool(jnp.isfinite(logqs).all())}")

        return logqs


    guide_params_initial_localised = init_localised_guide_params(images_labelled_localised[0])
    guide_logqs_initial_localised = check_initial_localised_guide_density(guide_params_initial_localised)

    guide_logqs_initial_localised
    return guide_params_initial_localised, localised_guide_log_prob


@app.cell
def _(
    guide_params_initial_localised,
    jax,
    jnp,
    jr,
    localised_guide_log_prob,
    optax,
    sample_labelled_tiles_localised,
    slice_tree_localised,
):
    NPE_BATCH_LOCALISED = 8
    NPE_STEPS_LOCALISED = 400
    NPE_LR_LOCALISED = 3e-4
    NPE_LOG_EVERY_LOCALISED = 25


    def npe_loss_localised(guide_params, theta_batch, images):
        losses = []

        for i in range(images.shape[0]):
            theta_i = slice_tree_localised(theta_batch, i)
            losses.append(-localised_guide_log_prob(guide_params, theta_i, images[i]))

        return jnp.mean(jnp.asarray(losses))


    def npe_train_step_localised(guide_params, opt_state, theta_batch, images):
        loss, grads = jax.value_and_grad(npe_loss_localised)(
            guide_params,
            theta_batch,
            images,
        )
        updates, opt_state = npe_optimiser_localised.update(grads, opt_state, guide_params)
        guide_params = optax.apply_updates(guide_params, updates)
        return guide_params, opt_state, loss


    npe_optimiser_localised = optax.adam(NPE_LR_LOCALISED)
    guide_params_npe_localised = guide_params_initial_localised
    npe_opt_state_localised = npe_optimiser_localised.init(guide_params_npe_localised)

    npe_key_localised = jr.PRNGKey(1001)
    npe_history_localised = []

    for step in range(NPE_STEPS_LOCALISED):
        npe_key_localised, batch_key = jr.split(npe_key_localised)

        theta_batch, image_batch, mean_batch, position_batch, sample_batch = (
            sample_labelled_tiles_localised(batch_key, n=NPE_BATCH_LOCALISED)
        )

        guide_params_npe_localised, npe_opt_state_localised, loss = (
            npe_train_step_localised(
                guide_params_npe_localised,
                npe_opt_state_localised,
                theta_batch,
                image_batch,
            )
        )

        if step % NPE_LOG_EVERY_LOCALISED == 0 or step == NPE_STEPS_LOCALISED - 1:
            loss_value = float(loss)
            npe_history_localised.append((step, loss_value))
            print(f"step {step:04d}  npe={loss_value:.2f}")


    npe_history_localised
    return


if __name__ == "__main__":
    app.run()
