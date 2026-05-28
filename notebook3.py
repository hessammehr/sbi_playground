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
    return COMPOSITION_BASIS, PIXELS, cfg, microscopy_model, render_latents


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
    Predictive,
    cfg,
    dist,
    jnp,
    jr,
    np,
    numpyro,
    plt,
    render_latents,
):
    def slot_grid_geometry(cfg=cfg):
        n = cfg.max_objects
        rows = int(np.floor(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        ys = (jnp.arange(rows) + 0.5) / rows
        xs = (jnp.arange(cols) + 0.5) / cols
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

        anchors01 = jnp.stack([yy.ravel(), xx.ravel()], axis=-1)[:n]

        cell_size = jnp.array(
            [
                cfg.tile_h / rows,
                cfg.tile_w / cols,
            ],
            dtype=jnp.float32,
        )

        anchors = anchors01 * jnp.array([cfg.tile_h, cfg.tile_w], dtype=jnp.float32)

        return rows, cols, anchors01, anchors, cell_size


    SLOT_GRID_ROWS, SLOT_GRID_COLS, SLOT_ANCHORS01, SLOT_ANCHORS, SLOT_CELL_SIZE = (
        slot_grid_geometry(cfg)
    )

    CELL_POSITION_OFFSET_SCALE = 0.28
    CELL_POSITION_OFFSET_LIMIT = 0.48


    def cellular_position_from_offset(position_offset, cfg=cfg):
        offset = jnp.clip(
            position_offset,
            -CELL_POSITION_OFFSET_LIMIT,
            CELL_POSITION_OFFSET_LIMIT,
        )
        return SLOT_ANCHORS + offset * SLOT_CELL_SIZE


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
                dist.Normal(0.0, CELL_POSITION_OFFSET_SCALE)
                .expand((2,))
                .to_event(1),
            )

            position = numpyro.deterministic(
                "position",
                cellular_position_from_offset(position_offset, cfg),
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


    def prior_predictive_localised(key=jr.PRNGKey(0), num_samples=4):
        return Predictive(
            microscopy_model_localised,
            num_samples=num_samples,
            return_sites=display_sites_localised,
        )(key)


    def show_cellular_slot_geometry():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(jnp.zeros((cfg.tile_h, cfg.tile_w, 3)))

        ax.scatter(
            np.asarray(SLOT_ANCHORS[:, 1]),
            np.asarray(SLOT_ANCHORS[:, 0]),
            s=30,
        )

        for idx, (y, x) in enumerate(np.asarray(SLOT_ANCHORS)):
            ax.text(x + 1, y + 1, str(idx), fontsize=8)

        for row_idx in range(1, SLOT_GRID_ROWS):
            y = row_idx * cfg.tile_h / SLOT_GRID_ROWS
            ax.axhline(y, linewidth=0.5)

        for col_idx in range(1, SLOT_GRID_COLS):
            x = col_idx * cfg.tile_w / SLOT_GRID_COLS
            ax.axvline(x, linewidth=0.5)

        ax.set_title("Cellular slot geometry")
        ax.set_xlim(0, cfg.tile_w)
        ax.set_ylim(cfg.tile_h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        return fig


    fig_cellular_slot_geometry = show_cellular_slot_geometry()
    fig_cellular_slot_geometry
    return (
        CELL_POSITION_OFFSET_SCALE,
        SLOT_ANCHORS,
        SLOT_ANCHORS01,
        cellular_position_from_offset,
        display_sites_localised,
        latent_sample_sites_localised,
        microscopy_model_localised,
        prior_predictive_localised,
    )


@app.cell
def _(
    Predictive,
    display_sites_localised,
    jax,
    jnp,
    jr,
    latent_sample_sites_localised,
    microscopy_model_localised,
    np,
    plt,
):
    @jax.jit(static_argnums=1)
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
        draw_objects_localised,
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
    return (
        GUIDE_MIN_CONC_LOCALISED,
        guide_positive_scale_localised,
        localised_amortised_guide,
    )


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


    @jax.jit
    def npe_loss_localised(guide_params, theta_batch, images):
        losses = []

        for i in range(images.shape[0]):
            theta_i = slice_tree_localised(theta_batch, i)
            losses.append(-localised_guide_log_prob(guide_params, theta_i, images[i]))

        return jnp.mean(jnp.asarray(losses))

    @jax.jit
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
    return (guide_params_npe_localised,)


@app.cell
def _(
    Predictive,
    SLOT_ANCHORS01,
    cfg,
    draw_objects_localised,
    guide_params_npe_localised,
    jax,
    jnp,
    jr,
    localised_amortised_guide,
    np,
    numpyro,
    plt,
    sample_labelled_tiles_localised,
):
    def _logit(x, eps=1e-5):
        x = jnp.clip(x, eps, 1.0 - eps)
        return jnp.log(x) - jnp.log1p(-x)

    SLOT_ANCHOR_LOGITS = _logit(SLOT_ANCHORS01)

    def guide_predictive_localised(guide_params, image, num_samples=32, key=jr.PRNGKey(0)):
        predictive = Predictive(
            numpyro.handlers.substitute(localised_amortised_guide, data=guide_params),
            num_samples=num_samples,
            return_sites=[
                "presence",
                "position_offset",
                "size",
                "composition",
            ],
        )
        return predictive(key, image)


    def position_from_offset_localised(position_offset):
        position01 = jax.nn.sigmoid(SLOT_ANCHOR_LOGITS + position_offset)
        return position01 * jnp.array([cfg.tile_h, cfg.tile_w])


    def visualise_guide_recovery_localised(
        guide_params=guide_params_npe_localised,
        key=jr.PRNGKey(202),
        n=4,
        posterior_samples=64,
    ):
        theta, images, means, true_positions, samples = sample_labelled_tiles_localised(key, n=n)

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

        posterior_keys = jr.split(jr.fold_in(key, 1), n)

        for i in range(n):
            image = images[i]
            q_samples = guide_predictive_localised(
                guide_params,
                image,
                num_samples=posterior_samples,
                key=posterior_keys[i],
            )

            q_presence_prob = q_samples["presence"].mean(axis=0)
            q_position = position_from_offset_localised(q_samples["position_offset"]).mean(axis=0)
            q_size = q_samples["size"].mean(axis=0)
            q_comp = q_samples["composition"].mean(axis=0)

            true_presence = theta["presence"][i]
            true_position = true_positions[i]
            true_size = theta["size"][i]

            ax = axes[i, 0]
            ax.imshow(np.asarray(jnp.clip(image, 0.0, 1.0)))
            draw_objects_localised(ax, true_position, true_presence, true_size)
            ax.set_title("True latent objects")
            ax.axis("off")

            ax = axes[i, 1]
            ax.imshow(np.asarray(jnp.clip(image, 0.0, 1.0)))
            draw_objects_localised(ax, q_position, q_presence_prob > 0.5, q_size)
            ax.set_title("Guide posterior mean objects")
            ax.axis("off")

            ax = axes[i, 2]
            ax.imshow(np.asarray(jnp.clip(image, 0.0, 1.0)))

            q_presence_np = np.asarray(q_presence_prob)
            q_position_np = np.asarray(q_position)
            q_comp_np = np.asarray(q_comp)

            for p, (y, x), comp in zip(q_presence_np, q_position_np, q_comp_np):
                if p < 0.25:
                    continue
                ax.scatter(x, y, s=20 + 60 * p)
                ax.text(
                    x + 1,
                    y + 1,
                    f"{p:.2f}\n[{comp[0]:.1f},{comp[1]:.1f},{comp[2]:.1f}]",
                    fontsize=6,
                )

            ax.set_title("Guide presence + composition")
            ax.set_xlim(0, cfg.tile_w)
            ax.set_ylim(cfg.tile_h, 0)
            ax.axis("off")

        fig.suptitle("Synthetic recovery check after NPE pretraining", y=1.01)
        fig.tight_layout()

        return fig, theta, images, q_samples


    fig_guide_recovery_localised, theta_recovery_localised, images_recovery_localised, q_samples_recovery_localised = (
        visualise_guide_recovery_localised()
    )

    fig_guide_recovery_localised
    return (position_from_offset_localised,)


@app.cell
def _(
    GUIDE_MIN_CONC_LOCALISED,
    SLOT_ANCHORS,
    SLOT_ANCHORS01,
    cfg,
    dist,
    guide_positive_scale_localised,
    images_labelled_localised,
    jax,
    jnp,
    jr,
    linen,
    numpyro,
    numpyro_flax_module,
):
    GUIDE_SPATIAL_EMB_DIM_LOCALISED = 128
    GUIDE_SPATIAL_SLOT_DIM_LOCALISED = 96
    GUIDE_SPATIAL_HIDDEN_LOCALISED = 128

    SLOT_FEATURE_H_LOCALISED = cfg.tile_h // 8
    SLOT_FEATURE_W_LOCALISED = cfg.tile_w // 8

    SLOT_FEATURE_Y_LOCALISED = jnp.clip(
        jnp.round(SLOT_ANCHORS[:, 0] / cfg.tile_h * (SLOT_FEATURE_H_LOCALISED - 1)).astype(jnp.int32),
        0,
        SLOT_FEATURE_H_LOCALISED - 1,
    )

    SLOT_FEATURE_X_LOCALISED = jnp.clip(
        jnp.round(SLOT_ANCHORS[:, 1] / cfg.tile_w * (SLOT_FEATURE_W_LOCALISED - 1)).astype(jnp.int32),
        0,
        SLOT_FEATURE_W_LOCALISED - 1,
    )


    class LocalisedSpatialEncoder(linen.Module):
        global_dim: int = GUIDE_SPATIAL_EMB_DIM_LOCALISED
        slot_dim: int = GUIDE_SPATIAL_SLOT_DIM_LOCALISED

        @linen.compact
        def __call__(self, image):
            x = image[None, ...]

            for channels in (24, 48, 96):
                x = linen.Conv(channels, (3, 3), strides=(2, 2), padding="SAME")(x)
                x = linen.gelu(x)

            feature_map = x[0]  # (H/8, W/8, C)

            global_emb = feature_map.mean(axis=(0, 1))
            global_emb = linen.Dense(self.global_dim)(global_emb)
            global_emb = linen.gelu(global_emb)

            slot_features = feature_map[SLOT_FEATURE_Y_LOCALISED, SLOT_FEATURE_X_LOCALISED, :]

            slot_inputs = jnp.concatenate(
                [
                    slot_features,
                    SLOT_ANCHORS01,
                ],
                axis=-1,
            )

            slot_emb = linen.Dense(self.slot_dim)(slot_inputs)
            slot_emb = linen.gelu(slot_emb)
            slot_emb = linen.Dense(self.slot_dim)(slot_emb)
            slot_emb = linen.gelu(slot_emb)

            return global_emb, slot_emb


    class LocalisedSpatialPosteriorHead(linen.Module):
        composition_dim: int
        hidden: int = GUIDE_SPATIAL_HIDDEN_LOCALISED

        @linen.compact
        def __call__(self, global_emb, slot_emb):
            k = self.composition_dim

            h_global = linen.Dense(self.hidden)(global_emb)
            h_global = linen.gelu(h_global)
            h_global = linen.Dense(self.hidden)(h_global)
            h_global = linen.gelu(h_global)

            background = linen.Dense(2 * 4 * 3)(h_global).reshape(4, 3, 2)
            object_gain = linen.Dense(2)(h_global)
            obs_sigma = linen.Dense(2)(h_global)

            h_slot = linen.Dense(self.hidden)(slot_emb)
            h_slot = linen.gelu(h_slot)
            h_slot = linen.Dense(self.hidden)(h_slot)
            h_slot = linen.gelu(h_slot)

            presence_logits = linen.Dense(1)(h_slot).squeeze(-1)
            position_offset = linen.Dense(4)(h_slot).reshape(cfg.max_objects, 2, 2)
            log_size = linen.Dense(2)(h_slot)
            composition_raw = linen.Dense(k)(h_slot)

            return {
                "background": background,
                "object_gain": object_gain,
                "obs_sigma": obs_sigma,
                "presence_logits": presence_logits,
                "position_offset": position_offset,
                "log_size": log_size,
                "composition_raw": composition_raw,
            }


    localised_spatial_encoder_module = LocalisedSpatialEncoder()

    localised_spatial_posterior_head_module = LocalisedSpatialPosteriorHead(
        composition_dim=cfg.composition_dim,
    )


    def localised_spatial_amortised_guide(image, cfg=cfg):
        encoder = numpyro_flax_module(
            "localised_spatial_encoder",
            localised_spatial_encoder_module,
            input_shape=(cfg.tile_h, cfg.tile_w, 3),
        )

        posterior_head = numpyro_flax_module(
            "localised_spatial_posterior_head",
            localised_spatial_posterior_head_module,
            jnp.zeros((GUIDE_SPATIAL_EMB_DIM_LOCALISED,)),
            jnp.zeros((cfg.max_objects, GUIDE_SPATIAL_SLOT_DIM_LOCALISED)),
        )

        global_emb, slot_emb = encoder(image)
        q = posterior_head(global_emb, slot_emb)

        bg_loc = q["background"][..., 0]
        bg_scale = guide_positive_scale_localised(q["background"][..., 1])
        numpyro.sample("background_logits", dist.Normal(bg_loc, bg_scale).to_event(2))

        object_gain_loc = q["object_gain"][0]
        object_gain_scale = guide_positive_scale_localised(q["object_gain"][1])
        numpyro.sample("object_gain", dist.LogNormal(object_gain_loc, object_gain_scale))

        obs_sigma_loc = q["obs_sigma"][0]
        obs_sigma_scale = guide_positive_scale_localised(q["obs_sigma"][1])
        numpyro.sample("obs_sigma", dist.LogNormal(obs_sigma_loc, obs_sigma_scale))

        with numpyro.plate("objects", cfg.max_objects):
            numpyro.sample("presence", dist.Bernoulli(logits=q["presence_logits"]))

            position_offset_loc = q["position_offset"][..., 0]
            position_offset_scale = guide_positive_scale_localised(q["position_offset"][..., 1])
            numpyro.sample(
                "position_offset",
                dist.Normal(position_offset_loc, position_offset_scale).to_event(1),
            )

            log_size_loc = q["log_size"][..., 0]
            log_size_scale = guide_positive_scale_localised(q["log_size"][..., 1])
            numpyro.sample("size", dist.LogNormal(log_size_loc, log_size_scale))

            composition_concentration = GUIDE_MIN_CONC_LOCALISED + jax.nn.softplus(
                q["composition_raw"]
            )
            numpyro.sample("composition", dist.Dirichlet(composition_concentration))


    def inspect_localised_spatial_guide_shapes(image=images_labelled_localised[0]):
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(localised_spatial_amortised_guide, jr.PRNGKey(0))
        ).get_trace(image)

        return {
            name: {
                "shape": tuple(site["value"].shape),
                "fn": type(site["fn"]).__name__,
            }
            for name, site in trace.items()
            if site["type"] == "sample"
        }


    guide_shape_summary_spatial_localised = inspect_localised_spatial_guide_shapes()
    guide_shape_summary_spatial_localised
    return (localised_spatial_amortised_guide,)


@app.cell
def _(
    images_labelled_localised,
    jax,
    jnp,
    jr,
    localised_spatial_amortised_guide,
    numpyro,
    optax,
    sample_labelled_tiles_localised,
    slice_tree_localised,
):
    SPATIAL_NPE_BATCH_LOCALISED = 8
    SPATIAL_NPE_STEPS_LOCALISED = 400
    SPATIAL_NPE_LR_LOCALISED = 3e-4
    SPATIAL_NPE_LOG_EVERY_LOCALISED = 25


    def init_localised_spatial_guide_params(image):
        seeded_guide = numpyro.handlers.seed(localised_spatial_amortised_guide, rng_seed=0)

        with numpyro.handlers.trace() as tr:
            seeded_guide(image)

        return {
            name: site["value"]
            for name, site in tr.items()
            if site["type"] == "param"
        }


    def trace_localised_spatial_guide_given_theta(guide_params, theta, image):
        values = {**guide_params, **theta}

        guide = numpyro.handlers.seed(localised_spatial_amortised_guide, rng_seed=0)
        guide = numpyro.handlers.substitute(guide, data=values)

        return numpyro.handlers.trace(guide).get_trace(image)

    @jax.jit
    def localised_spatial_guide_log_prob_masked(guide_params, theta, image):
        tr = trace_localised_spatial_guide_given_theta(guide_params, theta, image)
        presence = theta["presence"].astype(jnp.float32)

        total_logq = 0.0

        for site_name in ["background_logits", "object_gain", "obs_sigma", "presence"]:
            site = tr[site_name]
            total_logq = total_logq + jnp.sum(site["fn"].log_prob(site["value"]))

        for site_name in ["position_offset", "size", "composition"]:
            site = tr[site_name]
            per_slot_logq = site["fn"].log_prob(site["value"])
            total_logq = total_logq + jnp.sum(per_slot_logq * presence)

        return total_logq


    def spatial_npe_loss_localised(guide_params, theta_batch_arg, images_arg):
        per_example_losses = []

        for example_idx in range(images_arg.shape[0]):
            theta_i = slice_tree_localised(theta_batch_arg, example_idx)
            per_example_losses.append(
                -localised_spatial_guide_log_prob_masked(
                    guide_params,
                    theta_i,
                    images_arg[example_idx],
                )
            )

        return jnp.mean(jnp.asarray(per_example_losses))

    def spatial_npe_train_step_localised(guide_params, opt_state, theta_batch_arg, images_arg, optimiser):
        train_loss, grads = jax.value_and_grad(spatial_npe_loss_localised)(
            guide_params,
            theta_batch_arg,
            images_arg,
        )
        updates, opt_state = optimiser.update(grads, opt_state, guide_params)
        guide_params = optax.apply_updates(guide_params, updates)
        return guide_params, opt_state, train_loss


    def run_spatial_npe_training_localised(initial_params, key):
        optimiser = optax.adam(SPATIAL_NPE_LR_LOCALISED)
        opt_state = optimiser.init(initial_params)
        guide_params = initial_params
        history = []

        for train_step in range(SPATIAL_NPE_STEPS_LOCALISED):
            key, synthetic_key = jr.split(key)

            theta_synth, images_synth, _, _, _ = sample_labelled_tiles_localised(
                synthetic_key,
                n=SPATIAL_NPE_BATCH_LOCALISED,
            )

            guide_params, opt_state, train_loss = spatial_npe_train_step_localised(
                guide_params,
                opt_state,
                theta_synth,
                images_synth,
                optimiser,
            )

            if train_step % SPATIAL_NPE_LOG_EVERY_LOCALISED == 0 or train_step == SPATIAL_NPE_STEPS_LOCALISED - 1:
                train_loss_value = float(train_loss)
                history.append((train_step, train_loss_value))
                print(f"step {train_step:04d}  masked spatial npe={train_loss_value:.2f}")

        return guide_params, opt_state, key, history


    guide_params_spatial_initial_localised = init_localised_spatial_guide_params(
        images_labelled_localised[0]
    )

    (
        guide_params_spatial_npe_localised,
        spatial_npe_opt_state_localised,
        spatial_npe_key_localised,
        spatial_npe_history_localised,
    ) = run_spatial_npe_training_localised(
        guide_params_spatial_initial_localised,
        jr.PRNGKey(3001),
    )

    spatial_npe_history_localised
    return (
        guide_params_spatial_npe_localised,
        localised_spatial_guide_log_prob_masked,
    )


@app.cell
def _(
    Predictive,
    cfg,
    draw_objects_localised,
    guide_params_spatial_npe_localised,
    jnp,
    jr,
    localised_spatial_amortised_guide,
    np,
    numpyro,
    plt,
    position_from_offset_localised,
    sample_labelled_tiles_localised,
):
    def spatial_guide_predictive_localised(
        guide_params,
        image,
        num_samples=64,
        key=jr.PRNGKey(0),
    ):
        predictive = Predictive(
            numpyro.handlers.substitute(
                localised_spatial_amortised_guide,
                data=guide_params,
            ),
            num_samples=num_samples,
            return_sites=[
                "presence",
                "position_offset",
                "size",
                "composition",
            ],
        )
        return predictive(key, image)


    def summarise_spatial_guide_samples_localised(q_samples):
        presence_prob = q_samples["presence"].mean(axis=0)
        position_mean = position_from_offset_localised(q_samples["position_offset"]).mean(axis=0)
        size_mean = q_samples["size"].mean(axis=0)
        composition_mean = q_samples["composition"].mean(axis=0)

        return presence_prob, position_mean, size_mean, composition_mean


    def visualise_spatial_guide_recovery_localised(
        guide_params,
        key=jr.PRNGKey(404),
        n_examples=4,
        n_posterior_samples=64,
    ):
        theta_eval, images_eval, _, true_positions_eval, _ = sample_labelled_tiles_localised(
            key,
            n=n_examples,
        )

        posterior_keys = jr.split(jr.fold_in(key, 1), n_examples)

        fig, axes = plt.subplots(
            n_examples,
            4,
            figsize=(12, 3 * n_examples),
            squeeze=False,
        )

        position_errors = []
        count_errors = []

        for example_idx in range(n_examples):
            image_i = images_eval[example_idx]
            true_presence_i = theta_eval["presence"][example_idx]
            true_position_i = true_positions_eval[example_idx]
            true_size_i = theta_eval["size"][example_idx]
            true_comp_i = theta_eval["composition"][example_idx]

            q_samples_i = spatial_guide_predictive_localised(
                guide_params,
                image_i,
                num_samples=n_posterior_samples,
                key=posterior_keys[example_idx],
            )

            q_presence_i, q_position_i, q_size_i, q_comp_i = (
                summarise_spatial_guide_samples_localised(q_samples_i)
            )

            true_present_mask_i = true_presence_i.astype(bool)
            inferred_present_mask_i = q_presence_i > 0.5

            if true_present_mask_i.sum() > 0:
                slot_position_error_i = jnp.linalg.norm(
                    q_position_i[true_present_mask_i] - true_position_i[true_present_mask_i],
                    axis=-1,
                ).mean()
                position_errors.append(slot_position_error_i)

            count_errors.append(
                jnp.abs(q_presence_i.sum() - true_presence_i.sum())
            )

            ax = axes[example_idx, 0]
            ax.imshow(np.asarray(jnp.clip(image_i, 0.0, 1.0)))
            draw_objects_localised(ax, true_position_i, true_presence_i, true_size_i)
            ax.set_title(f"True objects\nN={int(true_presence_i.sum())}")
            ax.axis("off")

            ax = axes[example_idx, 1]
            ax.imshow(np.asarray(jnp.clip(image_i, 0.0, 1.0)))
            draw_objects_localised(ax, q_position_i, inferred_present_mask_i, q_size_i)
            ax.set_title(f"Spatial guide mean\nE[N]={float(q_presence_i.sum()):.1f}")
            ax.axis("off")

            ax = axes[example_idx, 2]
            ax.imshow(np.asarray(jnp.clip(image_i, 0.0, 1.0)))

            for p_i, pos_i, comp_i in zip(
                np.asarray(q_presence_i),
                np.asarray(q_position_i),
                np.asarray(q_comp_i),
            ):
                if p_i < 0.25:
                    continue

                y_i, x_i = pos_i
                ax.scatter(x_i, y_i, s=20 + 70 * p_i)
                ax.text(
                    x_i + 1,
                    y_i + 1,
                    f"{p_i:.2f}\n[{comp_i[0]:.1f},{comp_i[1]:.1f},{comp_i[2]:.1f}]",
                    fontsize=6,
                )

            ax.set_title("Presence + composition")
            ax.set_xlim(0, cfg.tile_w)
            ax.set_ylim(cfg.tile_h, 0)
            ax.axis("off")

            ax = axes[example_idx, 3]
            ax.scatter(
                np.asarray(true_position_i[:, 1]),
                np.asarray(true_position_i[:, 0]),
                s=20 + 80 * np.asarray(true_presence_i),
                label="true",
            )
            ax.scatter(
                np.asarray(q_position_i[:, 1]),
                np.asarray(q_position_i[:, 0]),
                s=20 + 80 * np.asarray(q_presence_i),
                marker="x",
                label="guide",
            )
            ax.set_title("Slotwise positions")
            ax.set_xlim(0, cfg.tile_w)
            ax.set_ylim(cfg.tile_h, 0)
            ax.set_aspect("equal")
            ax.legend(fontsize=7)

        mean_position_error = (
            float(jnp.asarray(position_errors).mean())
            if position_errors
            else float("nan")
        )
        mean_count_error = float(jnp.asarray(count_errors).mean())

        fig.suptitle(
            f"Spatial-guide synthetic recovery: "
            f"position MAE≈{mean_position_error:.2f}px, "
            f"count error≈{mean_count_error:.2f}",
            y=1.01,
        )
        fig.tight_layout()

        return fig


    fig_spatial_guide_recovery_localised = visualise_spatial_guide_recovery_localised(
        guide_params_spatial_npe_localised,
    )

    fig_spatial_guide_recovery_localised
    return (visualise_spatial_guide_recovery_localised,)


@app.cell
def _(
    guide_params_spatial_npe_localised,
    jax,
    jnp,
    jr,
    localised_spatial_amortised_guide,
    localised_spatial_guide_log_prob_masked,
    numpyro,
    optax,
    position_from_offset_localised,
    sample_labelled_tiles_localised,
    slice_tree_localised,
):
    AUX_SPATIAL_NPE_BATCH_LOCALISED = 8
    AUX_SPATIAL_NPE_STEPS_LOCALISED = 600
    AUX_SPATIAL_NPE_LR_LOCALISED = 3e-4
    AUX_SPATIAL_NPE_LOG_EVERY_LOCALISED = 25

    AUX_POSITION_WEIGHT_LOCALISED = 0.02
    AUX_PRESENCE_WEIGHT_LOCALISED = 5.0
    AUX_COMPOSITION_WEIGHT_LOCALISED = 1.0


    def spatial_guide_means_localised(guide_params, image):
        substituted = numpyro.handlers.substitute(
            localised_spatial_amortised_guide,
            data=guide_params,
        )
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(substituted, rng_seed=0)
        ).get_trace(image)

        presence_site = traced["presence"]
        position_site = traced["position_offset"]
        size_site = traced["size"]
        composition_site = traced["composition"]

        presence_prob = jax.nn.sigmoid(presence_site["fn"].logits)

        position_offset_mean = position_site["fn"].base_dist.loc
        position_mean = position_from_offset_localised(position_offset_mean)

        size_mean = jnp.exp(size_site["fn"].base_dist.loc)

        composition_conc = composition_site["fn"].concentration
        composition_mean = composition_conc / composition_conc.sum(axis=-1, keepdims=True)

        return presence_prob, position_mean, size_mean, composition_mean

    @jax.jit
    def auxiliary_supervised_loss_localised(guide_params, theta, image):
        q_presence, q_position, _, q_composition = spatial_guide_means_localised(
            guide_params,
            image,
        )

        true_presence = theta["presence"].astype(jnp.float32)
        true_position = position_from_offset_localised(theta["position_offset"])
        true_composition = theta["composition"]

        bce = optax.sigmoid_binary_cross_entropy(
            logits=jnp.log(q_presence + 1e-6) - jnp.log1p(-q_presence + 1e-6),
            labels=true_presence,
        ).mean()

        position_sqerr = jnp.sum((q_position - true_position) ** 2, axis=-1)
        position_loss = (position_sqerr * true_presence).sum() / (true_presence.sum() + 1e-6)

        composition_sqerr = jnp.sum((q_composition - true_composition) ** 2, axis=-1)
        composition_loss = (composition_sqerr * true_presence).sum() / (true_presence.sum() + 1e-6)

        return (
            AUX_PRESENCE_WEIGHT_LOCALISED * bce
            + AUX_POSITION_WEIGHT_LOCALISED * position_loss
            + AUX_COMPOSITION_WEIGHT_LOCALISED * composition_loss
        )

    @jax.jit
    def aux_spatial_npe_loss_localised(guide_params, theta_batch_arg, images_arg):
        per_example_losses = []

        for example_idx in range(images_arg.shape[0]):
            theta_i = slice_tree_localised(theta_batch_arg, example_idx)
            npe_i = -localised_spatial_guide_log_prob_masked(
                guide_params,
                theta_i,
                images_arg[example_idx],
            )
            aux_i = auxiliary_supervised_loss_localised(
                guide_params,
                theta_i,
                images_arg[example_idx],
            )
            per_example_losses.append(npe_i + aux_i)

        return jnp.mean(jnp.asarray(per_example_losses))


    def aux_spatial_npe_train_step_localised(guide_params, opt_state, theta_batch_arg, images_arg, optimiser):
        train_loss, grads = jax.value_and_grad(aux_spatial_npe_loss_localised)(
            guide_params,
            theta_batch_arg,
            images_arg,
        )
        updates, opt_state = optimiser.update(grads, opt_state, guide_params)
        guide_params = optax.apply_updates(guide_params, updates)
        return guide_params, opt_state, train_loss


    def run_aux_spatial_npe_training_localised(initial_params, key):
        optimiser = optax.adam(AUX_SPATIAL_NPE_LR_LOCALISED)
        opt_state = optimiser.init(initial_params)
        guide_params = initial_params
        history = []

        for train_step in range(AUX_SPATIAL_NPE_STEPS_LOCALISED):
            key, synthetic_key = jr.split(key)

            theta_synth, images_synth, _, _, _ = sample_labelled_tiles_localised(
                synthetic_key,
                n=AUX_SPATIAL_NPE_BATCH_LOCALISED,
            )

            guide_params, opt_state, train_loss = aux_spatial_npe_train_step_localised(
                guide_params,
                opt_state,
                theta_synth,
                images_synth,
                optimiser,
            )

            if train_step % AUX_SPATIAL_NPE_LOG_EVERY_LOCALISED == 0 or train_step == AUX_SPATIAL_NPE_STEPS_LOCALISED - 1:
                train_loss_value = float(train_loss)
                history.append((train_step, train_loss_value))
                print(f"step {train_step:04d}  aux spatial npe={train_loss_value:.2f}")

        return guide_params, opt_state, key, history


    (
        guide_params_aux_spatial_npe_localised,
        aux_spatial_npe_opt_state_localised,
        aux_spatial_npe_key_localised,
        aux_spatial_npe_history_localised,
    ) = run_aux_spatial_npe_training_localised(
        guide_params_spatial_npe_localised,
        jr.PRNGKey(5001),
    )

    aux_spatial_npe_history_localised
    return (guide_params_aux_spatial_npe_localised,)


@app.cell
def _(
    guide_params_aux_spatial_npe_localised,
    visualise_spatial_guide_recovery_localised,
):
    visualise_spatial_guide_recovery_localised(
        guide_params_aux_spatial_npe_localised,
    )
    return


@app.cell
def _(
    CELL_POSITION_OFFSET_SCALE,
    PIXELS,
    SLOT_ANCHORS,
    SLOT_ANCHORS01,
    cellular_position_from_offset,
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
    OBJECT_GUIDE_CROP_SIZE = 65
    OBJECT_GUIDE_GLOBAL_DIM = 128
    OBJECT_GUIDE_SLOT_DIM = 128
    OBJECT_GUIDE_HIDDEN = 128

    OBJECT_GUIDE_MIN_SCALE = 1e-3
    OBJECT_GUIDE_MAX_SCALE = 2.0
    OBJECT_GUIDE_MIN_CONCENTRATION = 1e-2


    def object_guide_scale(raw, min_value=OBJECT_GUIDE_MIN_SCALE, max_value=OBJECT_GUIDE_MAX_SCALE):
        return min_value + (max_value - min_value) * jax.nn.sigmoid(raw)


    def object_guide_crop_starts(cfg=cfg):
        half = OBJECT_GUIDE_CROP_SIZE // 2
        y0 = jnp.round(SLOT_ANCHORS[:, 0]).astype(jnp.int32) - half
        x0 = jnp.round(SLOT_ANCHORS[:, 1]).astype(jnp.int32) - half

        y0 = jnp.clip(y0, 0, cfg.tile_h - OBJECT_GUIDE_CROP_SIZE)
        x0 = jnp.clip(x0, 0, cfg.tile_w - OBJECT_GUIDE_CROP_SIZE)

        return jnp.stack([y0, x0], axis=-1)


    OBJECT_GUIDE_CROP_STARTS = object_guide_crop_starts(cfg)


    def object_guide_slot_crops(image):
        crop_half = OBJECT_GUIDE_CROP_SIZE / 2.0

        def crop_one(start_yx, anchor_yx):
            image_crop = jax.lax.dynamic_slice(
                image,
                (start_yx[0], start_yx[1], 0),
                (OBJECT_GUIDE_CROP_SIZE, OBJECT_GUIDE_CROP_SIZE, 3),
            )

            pixel_crop = jax.lax.dynamic_slice(
                PIXELS,
                (start_yx[0], start_yx[1], 0),
                (OBJECT_GUIDE_CROP_SIZE, OBJECT_GUIDE_CROP_SIZE, 2),
            )

            rel_yx = (pixel_crop - anchor_yx[None, None, :]) / crop_half
            return jnp.concatenate([image_crop, rel_yx], axis=-1)

        return jax.vmap(crop_one)(OBJECT_GUIDE_CROP_STARTS, SLOT_ANCHORS)


    def object_guide_position_from_offset(position_offset):
        return cellular_position_from_offset(position_offset, cfg)
        # position01 = jax.nn.sigmoid(SLOT_ANCHOR_LOGITS + position_offset)
        # return position01 * jnp.array([cfg.tile_h, cfg.tile_w])


    class ObjectSlotPatchGuide(linen.Module):
        max_objects: int
        composition_dim: int
        global_dim: int = OBJECT_GUIDE_GLOBAL_DIM
        slot_dim: int = OBJECT_GUIDE_SLOT_DIM
        hidden: int = OBJECT_GUIDE_HIDDEN

        @linen.compact
        def __call__(self, image):
            full = image[None, ...]

            for channels in (24, 48, 96):
                full = linen.Conv(channels, (3, 3), strides=(2, 2), padding="SAME")(full)
                full = linen.gelu(full)

            global_emb = full.mean(axis=(1, 2))[0]
            global_emb = linen.Dense(self.global_dim)(global_emb)
            global_emb = linen.gelu(global_emb)

            global_head = linen.Dense(self.hidden)(global_emb)
            global_head = linen.gelu(global_head)
            global_head = linen.Dense(self.hidden)(global_head)
            global_head = linen.gelu(global_head)

            background = linen.Dense(2 * 4 * 3)(global_head).reshape(4, 3, 2)
            object_gain = linen.Dense(2)(global_head)
            obs_sigma = linen.Dense(2)(global_head)

            slot = object_guide_slot_crops(image)

            for channels in (24, 48, 96, 128):
                slot = linen.Conv(channels, (3, 3), strides=(2, 2), padding="SAME")(slot)
                slot = linen.gelu(slot)

            slot = slot.mean(axis=(1, 2))

            global_per_slot = jnp.broadcast_to(
                global_emb[None, :],
                (self.max_objects, self.global_dim),
            )

            slot = jnp.concatenate(
                [
                    slot,
                    SLOT_ANCHORS01,
                    global_per_slot,
                ],
                axis=-1,
            )

            slot = linen.Dense(self.slot_dim)(slot)
            slot = linen.gelu(slot)
            slot = linen.Dense(self.slot_dim)(slot)
            slot = linen.gelu(slot)

            presence_logits = linen.Dense(1)(slot).squeeze(-1)
            position_offset = linen.Dense(4)(slot).reshape(self.max_objects, 2, 2)
            log_size = linen.Dense(2)(slot)
            composition_raw = linen.Dense(self.composition_dim)(slot)

            return {
                "background": background,
                "object_gain": object_gain,
                "obs_sigma": obs_sigma,
                "presence_logits": presence_logits,
                "position_offset": position_offset,
                "log_size": log_size,
                "composition_raw": composition_raw,
            }


    object_slot_patch_guide_module = ObjectSlotPatchGuide(
        max_objects=cfg.max_objects,
        composition_dim=cfg.composition_dim,
    )


    def object_slot_patch_amortised_guide(image, cfg=cfg):
        guide_net = numpyro_flax_module(
            "object_slot_patch_guide",
            object_slot_patch_guide_module,
            input_shape=(cfg.tile_h, cfg.tile_w, 3),
        )

        q = guide_net(image)

        bg_loc = q["background"][..., 0]
        bg_scale = object_guide_scale(q["background"][..., 1])
        numpyro.sample("background_logits", dist.Normal(bg_loc, bg_scale).to_event(2))

        object_gain_loc = q["object_gain"][0]
        object_gain_scale = object_guide_scale(q["object_gain"][1])
        numpyro.sample("object_gain", dist.LogNormal(object_gain_loc, object_gain_scale))

        obs_sigma_loc = q["obs_sigma"][0]
        obs_sigma_scale = object_guide_scale(q["obs_sigma"][1])
        numpyro.sample("obs_sigma", dist.LogNormal(obs_sigma_loc, obs_sigma_scale))

        with numpyro.plate("objects", cfg.max_objects):
            presence = numpyro.sample(
                "presence",
                dist.Bernoulli(logits=q["presence_logits"]),
            )
            present = presence[..., None].astype(jnp.float32)

            position_offset_loc_present = q["position_offset"][..., 0]
            position_offset_scale_present = object_guide_scale(q["position_offset"][..., 1])

            position_offset_loc = present * position_offset_loc_present
            position_offset_scale = (
                present * position_offset_scale_present
                + (1.0 - present) * CELL_POSITION_OFFSET_SCALE
            )

            numpyro.sample(
                "position_offset",
                dist.Normal(position_offset_loc, position_offset_scale).to_event(1),
            )

            log_size_loc_present = q["log_size"][..., 0]
            log_size_scale_present = object_guide_scale(q["log_size"][..., 1])

            present_flat = presence.astype(jnp.float32)
            log_size_loc = (
                present_flat * log_size_loc_present
                + (1.0 - present_flat) * cfg.size_loc
            )
            log_size_scale = (
                present_flat * log_size_scale_present
                + (1.0 - present_flat) * cfg.size_scale
            )

            numpyro.sample("size", dist.LogNormal(log_size_loc, log_size_scale))

            concentration_present = (
                OBJECT_GUIDE_MIN_CONCENTRATION
                + jax.nn.softplus(q["composition_raw"])
            )
            concentration_absent = jnp.ones_like(concentration_present)
            concentration = (
                present * concentration_present
                + (1.0 - present) * concentration_absent
            )

            numpyro.sample("composition", dist.Dirichlet(concentration))


    def inspect_object_slot_patch_guide_shapes(image=images_labelled_localised[0]):
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(object_slot_patch_amortised_guide, jr.PRNGKey(0))
        ).get_trace(image)

        return {
            name: {
                "shape": tuple(site["value"].shape),
                "fn": type(site["fn"]).__name__,
            }
            for name, site in trace.items()
            if site["type"] == "sample"
        }


    object_slot_patch_guide_shape_summary = inspect_object_slot_patch_guide_shapes()
    object_slot_patch_guide_shape_summary
    return (
        OBJECT_GUIDE_CROP_SIZE,
        object_guide_position_from_offset,
        object_slot_patch_amortised_guide,
    )


@app.cell
def _(
    cfg,
    images_labelled_localised,
    jax,
    jnp,
    jr,
    log_density,
    numpyro,
    object_slot_patch_amortised_guide,
    optax,
    sample_labelled_tiles_localised,
    slice_tree_localised,
):
    OBJECT_NPE_BATCH = 8
    OBJECT_NPE_STEPS = 600
    OBJECT_NPE_LR = 3e-4
    OBJECT_NPE_LOG_EVERY = 25


    def init_object_slot_patch_guide_params(image):
        seeded_guide = numpyro.handlers.seed(object_slot_patch_amortised_guide, rng_seed=0)

        with numpyro.handlers.trace() as tr:
            seeded_guide(image)

        return {
            name: site["value"]
            for name, site in tr.items()
            if site["type"] == "param"
        }

    @jax.jit
    def object_slot_patch_guide_log_prob(guide_params, theta, image):
        values = {**guide_params, **theta}

        logq, _ = log_density(
            object_slot_patch_amortised_guide,
            model_args=(image,),
            model_kwargs={"cfg": cfg},
            params=values,
        )

        return logq

    @jax.jit
    def object_npe_loss(guide_params, theta_batch_arg, images_arg):
        per_example_losses = []

        for example_idx in range(images_arg.shape[0]):
            theta_i = slice_tree_localised(theta_batch_arg, example_idx)
            per_example_losses.append(
                -object_slot_patch_guide_log_prob(
                    guide_params,
                    theta_i,
                    images_arg[example_idx],
                )
            )

        return jnp.mean(jnp.asarray(per_example_losses))


    def object_npe_train_step(guide_params, opt_state, theta_batch_arg, images_arg, optimiser):
        train_loss, grads = jax.value_and_grad(object_npe_loss)(
            guide_params,
            theta_batch_arg,
            images_arg,
        )
        updates, opt_state = optimiser.update(grads, opt_state, guide_params)
        guide_params = optax.apply_updates(guide_params, updates)
        return guide_params, opt_state, train_loss


    def run_object_npe_training(initial_params, key):
        optimiser = optax.adam(OBJECT_NPE_LR)
        opt_state = optimiser.init(initial_params)
        guide_params = initial_params
        history = []

        for train_step in range(OBJECT_NPE_STEPS):
            key, synthetic_key = jr.split(key)

            theta_synth, images_synth, _, _, _ = sample_labelled_tiles_localised(
                synthetic_key,
                n=OBJECT_NPE_BATCH,
            )

            guide_params, opt_state, train_loss = object_npe_train_step(
                guide_params,
                opt_state,
                theta_synth,
                images_synth,
                optimiser,
            )

            if train_step % OBJECT_NPE_LOG_EVERY == 0 or train_step == OBJECT_NPE_STEPS - 1:
                train_loss_value = float(train_loss)
                history.append((train_step, train_loss_value))
                print(f"step {train_step:04d}  object-slot NPE={train_loss_value:.2f}")

        return guide_params, opt_state, key, history


    object_slot_patch_guide_params_initial = init_object_slot_patch_guide_params(
        images_labelled_localised[0]
    )

    (
        object_slot_patch_guide_params_npe,
        object_slot_patch_npe_opt_state,
        object_slot_patch_npe_key,
        object_slot_patch_npe_history,
    ) = run_object_npe_training(
        object_slot_patch_guide_params_initial,
        jr.PRNGKey(7001),
    )

    object_slot_patch_npe_history
    return (
        init_object_slot_patch_guide_params,
        object_npe_loss,
        object_slot_patch_guide_params_initial,
        object_slot_patch_guide_params_npe,
    )


@app.cell
def _(
    Predictive,
    cfg,
    draw_objects_localised,
    jnp,
    jr,
    np,
    numpyro,
    object_guide_position_from_offset,
    object_slot_patch_amortised_guide,
    object_slot_patch_guide_params_npe,
    plt,
    sample_labelled_tiles_localised,
):
    def object_slot_patch_guide_predictive(
        guide_params,
        image,
        num_samples=64,
        key=jr.PRNGKey(0),
    ):
        predictive = Predictive(
            numpyro.handlers.substitute(
                object_slot_patch_amortised_guide,
                data=guide_params,
            ),
            num_samples=num_samples,
            return_sites=[
                "presence",
                "position_offset",
                "size",
                "composition",
            ],
        )
        return predictive(key, image)


    def summarise_object_slot_patch_samples(q_samples):
        presence_prob = q_samples["presence"].mean(axis=0)
        position_mean = object_guide_position_from_offset(
            q_samples["position_offset"]
        ).mean(axis=0)
        size_mean = q_samples["size"].mean(axis=0)
        composition_mean = q_samples["composition"].mean(axis=0)

        return presence_prob, position_mean, size_mean, composition_mean


    def visualise_object_slot_patch_recovery(
        guide_params,
        key=jr.PRNGKey(8001),
        n_examples=4,
        n_posterior_samples=64,
    ):
        theta_eval, images_eval, _, true_positions_eval, _ = sample_labelled_tiles_localised(
            key,
            n=n_examples,
        )

        posterior_keys = jr.split(jr.fold_in(key, 1), n_examples)

        fig, axes = plt.subplots(
            n_examples,
            4,
            figsize=(12, 3 * n_examples),
            squeeze=False,
        )

        position_errors = []
        count_errors = []
        composition_errors = []

        for example_idx in range(n_examples):
            image_i = images_eval[example_idx]
            true_presence_i = theta_eval["presence"][example_idx]
            true_position_i = true_positions_eval[example_idx]
            true_size_i = theta_eval["size"][example_idx]
            true_comp_i = theta_eval["composition"][example_idx]

            q_samples_i = object_slot_patch_guide_predictive(
                guide_params,
                image_i,
                num_samples=n_posterior_samples,
                key=posterior_keys[example_idx],
            )

            q_presence_i, q_position_i, q_size_i, q_comp_i = summarise_object_slot_patch_samples(
                q_samples_i
            )

            true_present_mask_i = true_presence_i.astype(bool)
            inferred_present_mask_i = q_presence_i > 0.5

            if true_present_mask_i.sum() > 0:
                position_errors.append(
                    jnp.linalg.norm(
                        q_position_i[true_present_mask_i]
                        - true_position_i[true_present_mask_i],
                        axis=-1,
                    ).mean()
                )
                composition_errors.append(
                    jnp.linalg.norm(
                        q_comp_i[true_present_mask_i]
                        - true_comp_i[true_present_mask_i],
                        axis=-1,
                    ).mean()
                )

            count_errors.append(jnp.abs(q_presence_i.sum() - true_presence_i.sum()))

            ax = axes[example_idx, 0]
            ax.imshow(np.asarray(jnp.clip(image_i, 0.0, 1.0)))
            draw_objects_localised(ax, true_position_i, true_presence_i, true_size_i)
            ax.set_title(f"True objects\nN={int(true_presence_i.sum())}")
            ax.axis("off")

            ax = axes[example_idx, 1]
            ax.imshow(np.asarray(jnp.clip(image_i, 0.0, 1.0)))
            draw_objects_localised(ax, q_position_i, inferred_present_mask_i, q_size_i)
            ax.set_title(f"Guide mean\nE[N]={float(q_presence_i.sum()):.1f}")
            ax.axis("off")

            ax = axes[example_idx, 2]
            ax.imshow(np.asarray(jnp.clip(image_i, 0.0, 1.0)))

            for p_i, pos_i, comp_i in zip(
                np.asarray(q_presence_i),
                np.asarray(q_position_i),
                np.asarray(q_comp_i),
            ):
                if p_i < 0.25:
                    continue

                y_i, x_i = pos_i
                ax.scatter(x_i, y_i, s=20 + 70 * p_i)
                ax.text(
                    x_i + 1,
                    y_i + 1,
                    f"{p_i:.2f}\n[{comp_i[0]:.1f},{comp_i[1]:.1f},{comp_i[2]:.1f}]",
                    fontsize=6,
                )

            ax.set_title("Presence + composition")
            ax.set_xlim(0, cfg.tile_w)
            ax.set_ylim(cfg.tile_h, 0)
            ax.axis("off")

            ax = axes[example_idx, 3]
            ax.scatter(
                np.asarray(true_position_i[:, 1]),
                np.asarray(true_position_i[:, 0]),
                s=20 + 80 * np.asarray(true_presence_i),
                label="true",
            )
            ax.scatter(
                np.asarray(q_position_i[:, 1]),
                np.asarray(q_position_i[:, 0]),
                s=20 + 80 * np.asarray(q_presence_i),
                marker="x",
                label="guide",
            )
            ax.set_title("Slotwise positions")
            ax.set_xlim(0, cfg.tile_w)
            ax.set_ylim(cfg.tile_h, 0)
            ax.set_aspect("equal")
            ax.legend(fontsize=7)

        mean_position_error = float(jnp.asarray(position_errors).mean())
        mean_count_error = float(jnp.asarray(count_errors).mean())
        mean_composition_error = float(jnp.asarray(composition_errors).mean())

        fig.suptitle(
            "Object-slot patch guide recovery: "
            f"position≈{mean_position_error:.2f}px, "
            f"count≈{mean_count_error:.2f}, "
            f"composition≈{mean_composition_error:.2f}",
            y=1.01,
        )
        fig.tight_layout()

        return fig


    fig_object_slot_patch_recovery = visualise_object_slot_patch_recovery(
        object_slot_patch_guide_params_npe,
    )

    fig_object_slot_patch_recovery
    return (
        object_slot_patch_guide_predictive,
        summarise_object_slot_patch_samples,
    )


@app.cell
def _(
    SLOT_ANCHORS,
    jnp,
    jr,
    np,
    object_slot_patch_guide_params_npe,
    object_slot_patch_guide_predictive,
    plt,
    sample_labelled_tiles_localised,
    summarise_object_slot_patch_samples,
):
    def object_slot_anchor_baseline_positions():
        return SLOT_ANCHORS


    def diagnose_object_slot_patch_against_anchors(
        guide_params,
        key=jr.PRNGKey(9001),
        n_examples=16,
        n_posterior_samples=64,
    ):
        theta_eval_diag, images_eval_diag, _, true_positions_eval_diag, _ = (
            sample_labelled_tiles_localised(key, n=n_examples)
        )

        posterior_keys_diag = jr.split(jr.fold_in(key, 1), n_examples)
        anchor_positions_diag = object_slot_anchor_baseline_positions()

        guide_position_errors_diag = []
        anchor_position_errors_diag = []
        presence_probs_diag = []
        true_presence_diag = []

        for example_idx_diag in range(n_examples):
            q_samples_diag = object_slot_patch_guide_predictive(
                guide_params,
                images_eval_diag[example_idx_diag],
                num_samples=n_posterior_samples,
                key=posterior_keys_diag[example_idx_diag],
            )

            q_presence_diag, q_position_diag, _, _ = summarise_object_slot_patch_samples(
                q_samples_diag
            )

            present_mask_diag = theta_eval_diag["presence"][example_idx_diag].astype(bool)

            if present_mask_diag.sum() > 0:
                guide_position_errors_diag.append(
                    jnp.linalg.norm(
                        q_position_diag[present_mask_diag]
                        - true_positions_eval_diag[example_idx_diag][present_mask_diag],
                        axis=-1,
                    )
                )
                anchor_position_errors_diag.append(
                    jnp.linalg.norm(
                        anchor_positions_diag[present_mask_diag]
                        - true_positions_eval_diag[example_idx_diag][present_mask_diag],
                        axis=-1,
                    )
                )

            presence_probs_diag.append(q_presence_diag)
            true_presence_diag.append(theta_eval_diag["presence"][example_idx_diag])

        guide_position_errors_diag = jnp.concatenate(guide_position_errors_diag)
        anchor_position_errors_diag = jnp.concatenate(anchor_position_errors_diag)

        presence_probs_diag = jnp.concatenate(presence_probs_diag)
        true_presence_diag = jnp.concatenate(true_presence_diag).astype(jnp.float32)

        presence_brier_diag = jnp.mean((presence_probs_diag - true_presence_diag) ** 2)
        presence_mean_present_diag = jnp.mean(presence_probs_diag[true_presence_diag > 0.5])
        presence_mean_absent_diag = jnp.mean(presence_probs_diag[true_presence_diag < 0.5])

        print("Position recovery on true-present slots only")
        print(f"guide MAE:  mean={guide_position_errors_diag.mean():.2f}px, median={jnp.median(guide_position_errors_diag):.2f}px")
        print(f"anchor MAE: mean={anchor_position_errors_diag.mean():.2f}px, median={jnp.median(anchor_position_errors_diag):.2f}px")
        print()
        print("Presence calibration")
        print(f"Brier score:       {presence_brier_diag:.3f}")
        print(f"E[q(present)|true present]: {presence_mean_present_diag:.3f}")
        print(f"E[q(present)|true absent]:  {presence_mean_absent_diag:.3f}")

        fig_diag, axes_diag = plt.subplots(1, 3, figsize=(12, 3.5))

        axes_diag[0].hist(np.asarray(anchor_position_errors_diag), bins=30, alpha=0.6, label="anchor")
        axes_diag[0].hist(np.asarray(guide_position_errors_diag), bins=30, alpha=0.6, label="guide")
        axes_diag[0].set_title("Position error on true-present slots")
        axes_diag[0].set_xlabel("error / px")
        axes_diag[0].legend()

        axes_diag[1].hist(
            np.asarray(presence_probs_diag[true_presence_diag < 0.5]),
            bins=30,
            alpha=0.6,
            label="true absent",
        )
        axes_diag[1].hist(
            np.asarray(presence_probs_diag[true_presence_diag > 0.5]),
            bins=30,
            alpha=0.6,
            label="true present",
        )
        axes_diag[1].set_title("Guide presence probabilities")
        axes_diag[1].set_xlabel("q(present | x)")
        axes_diag[1].legend()

        axes_diag[2].scatter(
            np.asarray(anchor_position_errors_diag),
            np.asarray(guide_position_errors_diag),
            s=12,
        )
        axes_diag[2].plot([0, 60], [0, 60], linestyle="--")
        axes_diag[2].set_xlim(0, 60)
        axes_diag[2].set_ylim(0, 60)
        axes_diag[2].set_xlabel("anchor error / px")
        axes_diag[2].set_ylabel("guide error / px")
        axes_diag[2].set_title("Does the guide beat anchors?")

        fig_diag.tight_layout()
        return fig_diag


    fig_object_slot_anchor_diagnostic = diagnose_object_slot_patch_against_anchors(
        object_slot_patch_guide_params_npe,
    )

    fig_object_slot_anchor_diagnostic
    return


@app.cell
def _(
    OBJECT_GUIDE_CROP_SIZE,
    SLOT_ANCHORS,
    jnp,
    jr,
    np,
    plt,
    sample_labelled_tiles_localised,
):
    def diagnose_anchor_crop_geometry(key=jr.PRNGKey(9101), n_examples=128):
        theta_diag, _, _, true_positions_diag, _ = sample_labelled_tiles_localised(
            key,
            n=n_examples,
        )

        present = theta_diag["presence"].astype(bool)
        true_positions = true_positions_diag
        anchor_positions = SLOT_ANCHORS[None, :, :]

        anchor_dist = jnp.linalg.norm(true_positions - anchor_positions, axis=-1)
        present_anchor_dist = anchor_dist[present]

        half_crop = OBJECT_GUIDE_CROP_SIZE / 2.0
        inside_current_crop = present_anchor_dist <= half_crop

        print("Anchor/crop geometry for true-present objects")
        print(f"present objects: {present_anchor_dist.shape[0]}")
        print(f"anchor distance mean:   {present_anchor_dist.mean():.2f}px")
        print(f"anchor distance median: {jnp.median(present_anchor_dist):.2f}px")
        print(f"anchor distance p90:    {jnp.quantile(present_anchor_dist, 0.90):.2f}px")
        print(f"current crop half-size: {half_crop:.1f}px")
        print(f"inside current crop:    {inside_current_crop.mean():.3f}")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(np.asarray(present_anchor_dist), bins=40)
        ax.axvline(half_crop, linestyle="--", label="current half crop")
        ax.set_title("Distance from true object to its slot anchor")
        ax.set_xlabel("distance / px")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()

        return fig


    fig_anchor_crop_geometry = diagnose_anchor_crop_geometry()
    fig_anchor_crop_geometry
    return


@app.cell
def _(
    SLOT_ANCHORS,
    cfg,
    draw_objects_localised,
    init_object_slot_patch_guide_params,
    jax,
    jnp,
    jr,
    np,
    object_guide_position_from_offset,
    object_npe_loss,
    object_slot_patch_guide_predictive,
    optax,
    plt,
    sample_labelled_tiles_localised,
    slice_tree_localised,
):
    OBJECT_OVERFIT_STEPS = 800
    OBJECT_OVERFIT_LR = 1e-3
    OBJECT_OVERFIT_LOG_EVERY = 50


    def object_single_example_diagnostics(guide_params, theta_single, image_single, key):
        q_samples = object_slot_patch_guide_predictive(
            guide_params,
            image_single,
            num_samples=256,
            key=key,
        )

        q_presence = q_samples["presence"].mean(axis=0)
        q_positions = object_guide_position_from_offset(q_samples["position_offset"])

        q_present_counts = q_samples["presence"].sum(axis=0)
        q_position_conditional = (
            (q_positions * q_samples["presence"][..., None]).sum(axis=0)
            / (q_present_counts[:, None] + 1e-6)
        )
        q_position_conditional = jnp.where(
            q_present_counts[:, None] > 0,
            q_position_conditional,
            SLOT_ANCHORS,
        )

        true_presence = theta_single["presence"].astype(jnp.float32)
        true_position = object_guide_position_from_offset(theta_single["position_offset"])

        present_mask = true_presence > 0.5
        absent_mask = true_presence < 0.5

        position_error = jnp.linalg.norm(
            q_position_conditional[present_mask] - true_position[present_mask],
            axis=-1,
        ).mean()

        present_prob = q_presence[present_mask].mean()
        absent_prob = q_presence[absent_mask].mean()

        return {
            "position_error": position_error,
            "present_prob": present_prob,
            "absent_prob": absent_prob,
            "q_presence": q_presence,
            "q_position": q_position_conditional,
        }


    def run_object_single_example_overfit(key):
        theta_one, images_one, _, _, _ = sample_labelled_tiles_localised(key, n=1)
        theta_single = slice_tree_localised(theta_one, 0)
        image_single = images_one[0]

        guide_params = init_object_slot_patch_guide_params(image_single)
        optimiser = optax.adam(OBJECT_OVERFIT_LR)
        opt_state = optimiser.init(guide_params)
        history = []

        def singleton_loss(params):
            return object_npe_loss(params, theta_one, images_one)

        for train_step in range(OBJECT_OVERFIT_STEPS):
            train_loss, grads = jax.value_and_grad(singleton_loss)(guide_params)
            updates, opt_state = optimiser.update(grads, opt_state, guide_params)
            guide_params = optax.apply_updates(guide_params, updates)

            if train_step % OBJECT_OVERFIT_LOG_EVERY == 0 or train_step == OBJECT_OVERFIT_STEPS - 1:
                metrics = object_single_example_diagnostics(
                    guide_params,
                    theta_single,
                    image_single,
                    jr.fold_in(key, train_step),
                )
                history.append(
                    (
                        train_step,
                        float(train_loss),
                        float(metrics["position_error"]),
                        float(metrics["present_prob"]),
                        float(metrics["absent_prob"]),
                    )
                )
                print(
                    f"step {train_step:04d}  "
                    f"loss={float(train_loss):.2f}  "
                    f"pos_err={float(metrics['position_error']):.2f}px  "
                    f"q_present={float(metrics['present_prob']):.2f}  "
                    f"q_absent={float(metrics['absent_prob']):.2f}"
                )

        final_metrics = object_single_example_diagnostics(
            guide_params,
            theta_single,
            image_single,
            jr.fold_in(key, 9999),
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        axes[0].imshow(np.asarray(jnp.clip(image_single, 0.0, 1.0)))
        draw_objects_localised(
            axes[0],
            true_position := object_guide_position_from_offset(theta_single["position_offset"]),
            theta_single["presence"],
            theta_single["size"],
        )
        axes[0].set_title(f"True objects\nN={int(theta_single['presence'].sum())}")
        axes[0].axis("off")

        axes[1].imshow(np.asarray(jnp.clip(image_single, 0.0, 1.0)))
        draw_objects_localised(
            axes[1],
            final_metrics["q_position"],
            final_metrics["q_presence"] > 0.5,
            theta_single["size"],
        )
        axes[1].set_title(
            "Overfit guide\n"
            f"q_present={float(final_metrics['present_prob']):.2f}, "
            f"q_absent={float(final_metrics['absent_prob']):.2f}"
        )
        axes[1].axis("off")

        axes[2].scatter(
            np.asarray(true_position[:, 1]),
            np.asarray(true_position[:, 0]),
            s=20 + 80 * np.asarray(theta_single["presence"]),
            label="true",
        )
        axes[2].scatter(
            np.asarray(final_metrics["q_position"][:, 1]),
            np.asarray(final_metrics["q_position"][:, 0]),
            s=20 + 80 * np.asarray(final_metrics["q_presence"]),
            marker="x",
            label="guide",
        )
        axes[2].set_xlim(0, cfg.tile_w)
        axes[2].set_ylim(cfg.tile_h, 0)
        axes[2].set_aspect("equal")
        axes[2].set_title(f"Single-example overfit\npos err={float(final_metrics['position_error']):.2f}px")
        axes[2].legend(fontsize=7)

        fig.tight_layout()

        return guide_params, history, fig


    object_overfit_guide_params, object_overfit_history, fig_object_single_overfit = (
        run_object_single_example_overfit(jr.PRNGKey(12345))
    )

    fig_object_single_overfit
    return (object_overfit_guide_params,)


@app.cell
def _(
    jnp,
    jr,
    numpyro,
    object_overfit_guide_params,
    object_slot_patch_amortised_guide,
    object_slot_patch_guide_params_initial,
    plt,
    sample_labelled_tiles_localised,
    slice_tree_localised,
):
    def object_slot_patch_site_log_probs(guide_params, theta, image):
        values = {**guide_params, **theta}

        guide = numpyro.handlers.seed(object_slot_patch_amortised_guide, rng_seed=0)
        guide = numpyro.handlers.substitute(guide, data=values)
        trace = numpyro.handlers.trace(guide).get_trace(image)

        site_logps = {}

        for site_name in [
            "background_logits",
            "object_gain",
            "obs_sigma",
            "presence",
            "position_offset",
            "size",
            "composition",
        ]:
            site = trace[site_name]
            lp = site["fn"].log_prob(site["value"])
            site_logps[site_name] = jnp.sum(lp)

        return site_logps


    def compare_site_log_probs_before_after_overfit(
        params_before=object_slot_patch_guide_params_initial,
        params_after=object_overfit_guide_params,
        key=jr.PRNGKey(12345),
    ):
        theta_one, images_one, _, _, _ = sample_labelled_tiles_localised(key, n=1)
        theta_single = slice_tree_localised(theta_one, 0)
        image_single = images_one[0]

        before = object_slot_patch_site_log_probs(params_before, theta_single, image_single)
        after = object_slot_patch_site_log_probs(params_after, theta_single, image_single)

        rows = []
        for site_name in before:
            rows.append(
                (
                    site_name,
                    float(before[site_name]),
                    float(after[site_name]),
                    float(after[site_name] - before[site_name]),
                )
            )

        print("Sitewise log q(theta | x) on the single overfit example")
        print("site                 before        after       delta")
        for site_name, before_lp, after_lp, delta_lp in rows:
            print(f"{site_name:18s} {before_lp:10.2f} {after_lp:10.2f} {delta_lp:10.2f}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels = [row[0] for row in rows]
        deltas = [row[3] for row in rows]
        ax.bar(labels, deltas)
        ax.axhline(0.0, linewidth=1)
        ax.set_ylabel("Δ log q")
        ax.set_title("Which guide sites improved during single-example overfit?")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        return fig, before, after


    fig_sitewise_overfit, site_logq_before_overfit, site_logq_after_overfit = (
        compare_site_log_probs_before_after_overfit()
    )

    fig_sitewise_overfit
    return


@app.cell
def _(
    cfg,
    jax,
    jnp,
    jr,
    np,
    numpyro,
    object_guide_position_from_offset,
    object_overfit_guide_params,
    object_slot_patch_amortised_guide,
    plt,
    sample_labelled_tiles_localised,
    slice_tree_localised,
):
    def object_slot_patch_distribution_params(guide_params, image):
        substituted = numpyro.handlers.substitute(
            object_slot_patch_amortised_guide,
            data=guide_params,
        )
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(substituted, rng_seed=0)
        ).get_trace(image)

        presence_fn = trace["presence"]["fn"]
        position_fn = trace["position_offset"]["fn"]
        size_fn = trace["size"]["fn"]
        composition_fn = trace["composition"]["fn"]

        presence_prob = jax.nn.sigmoid(presence_fn.logits)

        position_offset_loc = position_fn.base_dist.loc
        position_offset_scale = position_fn.base_dist.scale
        position_mean = object_guide_position_from_offset(position_offset_loc)

        log_size_loc = size_fn.base_dist.loc
        log_size_scale = size_fn.base_dist.scale
        size_mean = jnp.exp(log_size_loc)

        comp_conc = composition_fn.concentration
        comp_mean = comp_conc / comp_conc.sum(axis=-1, keepdims=True)

        return {
            "presence_prob": presence_prob,
            "position_offset_loc": position_offset_loc,
            "position_offset_scale": position_offset_scale,
            "position_mean": position_mean,
            "log_size_loc": log_size_loc,
            "log_size_scale": log_size_scale,
            "size_mean": size_mean,
            "composition_mean": comp_mean,
        }


    def diagnose_single_example_params_after_overfit(
        guide_params=object_overfit_guide_params,
        key=jr.PRNGKey(12345),
    ):
        theta_one, images_one, _, _, _ = sample_labelled_tiles_localised(key, n=1)
        theta_single = slice_tree_localised(theta_one, 0)
        image_single = images_one[0]

        q = object_slot_patch_distribution_params(guide_params, image_single)

        true_presence = theta_single["presence"].astype(jnp.float32)
        true_position_offset = theta_single["position_offset"]
        true_position = object_guide_position_from_offset(true_position_offset)
        true_size = theta_single["size"]

        present_mask = true_presence > 0.5
        absent_mask = true_presence < 0.5

        position_error = jnp.linalg.norm(
            q["position_mean"][present_mask] - true_position[present_mask],
            axis=-1,
        )

        offset_error = jnp.linalg.norm(
            q["position_offset_loc"][present_mask] - true_position_offset[present_mask],
            axis=-1,
        )

        print("Single-example guide parameter diagnostics")
        print(f"q(present) true-present mean: {q['presence_prob'][present_mask].mean():.3f}")
        print(f"q(present) true-absent  mean: {q['presence_prob'][absent_mask].mean():.3f}")
        print(f"position error mean:          {position_error.mean():.2f}px")
        print(f"offset error mean:            {offset_error.mean():.2f}")
        print(f"position offset scale mean:   {q['position_offset_scale'][present_mask].mean():.3f}")
        print(f"size true-present mean:       {true_size[present_mask].mean():.2f}")
        print(f"size q true-present mean:     {q['size_mean'][present_mask].mean():.2f}")
        print(f"log-size scale mean:          {q['log_size_scale'][present_mask].mean():.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        axes[0].scatter(
            np.asarray(true_position[:, 1]),
            np.asarray(true_position[:, 0]),
            s=20 + 80 * np.asarray(true_presence),
            label="true",
        )
        axes[0].scatter(
            np.asarray(q["position_mean"][:, 1]),
            np.asarray(q["position_mean"][:, 0]),
            s=20 + 80 * np.asarray(q["presence_prob"]),
            marker="x",
            label="guide mean",
        )
        axes[0].set_xlim(0, cfg.tile_w)
        axes[0].set_ylim(cfg.tile_h, 0)
        axes[0].set_aspect("equal")
        axes[0].set_title("Mean positions")
        axes[0].legend(fontsize=7)

        axes[1].scatter(
            np.asarray(true_position_offset[:, 1]),
            np.asarray(q["position_offset_loc"][:, 1]),
            s=20 + 80 * np.asarray(true_presence),
        )
        axes[1].plot([-3, 3], [-3, 3], linestyle="--")
        axes[1].set_xlim(-3, 3)
        axes[1].set_ylim(-3, 3)
        axes[1].set_xlabel("true x offset")
        axes[1].set_ylabel("q mean x offset")
        axes[1].set_title("Offset regression")

        axes[2].hist(
            np.asarray(q["presence_prob"][absent_mask]),
            bins=20,
            alpha=0.6,
            label="true absent",
        )
        axes[2].hist(
            np.asarray(q["presence_prob"][present_mask]),
            bins=20,
            alpha=0.6,
            label="true present",
        )
        axes[2].set_title("Presence probabilities")
        axes[2].legend(fontsize=7)

        fig.tight_layout()
        return fig, q, theta_single, image_single


    fig_single_param_diag, q_single_param_diag, theta_single_param_diag, image_single_param_diag = (
        diagnose_single_example_params_after_overfit()
    )

    fig_single_param_diag
    return


@app.cell
def _(
    PIXELS,
    SLOT_ANCHORS,
    cfg,
    jax,
    jnp,
    jr,
    np,
    plt,
    sample_labelled_tiles_localised,
):
    def simple_blob_score_image(image):
        intensity = image.mean(axis=-1)

        # Cheap background removal: subtract a broad local mean.
        # This is only a diagnostic, not part of the model.
        kernel_radius = 7
        padded = jnp.pad(intensity, ((kernel_radius, kernel_radius), (kernel_radius, kernel_radius)), mode="edge")

        def local_mean_at(yx):
            y, x = yx
            patch = jax.lax.dynamic_slice(
                padded,
                (y, x),
                (2 * kernel_radius + 1, 2 * kernel_radius + 1),
            )
            return patch.mean()

        coords = jnp.stack(
            jnp.meshgrid(
                jnp.arange(cfg.tile_h, dtype=jnp.int32),
                jnp.arange(cfg.tile_w, dtype=jnp.int32),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)

        local_mean = jax.vmap(local_mean_at)(coords).reshape(cfg.tile_h, cfg.tile_w)
        residual = jnp.clip(intensity - local_mean, 0.0, None)

        return residual


    def matched_filter_slot_estimates(image, search_radius=28.0, temperature=0.035):
        residual = simple_blob_score_image(image)

        delta = PIXELS[None, :, :, :] - SLOT_ANCHORS[:, None, None, :]
        r2 = jnp.sum(delta**2, axis=-1)
        window = jnp.exp(-0.5 * r2 / (search_radius**2))

        score = residual[None, :, :] * window
        mass = score.sum(axis=(1, 2)) + 1e-6

        position = (score[..., None] * PIXELS[None, :, :, :]).sum(axis=(1, 2)) / mass[:, None]

        # Not calibrated: only useful for ranking/separation.
        presence_score = mass / (window.sum(axis=(1, 2)) + 1e-6)
        presence_prob = jax.nn.sigmoid((presence_score - presence_score.mean()) / temperature)

        return presence_prob, position, residual


    def diagnose_matched_filter_baseline(key=jr.PRNGKey(9901), n_examples=64):
        theta_eval, images_eval, _, true_positions_eval, _ = sample_labelled_tiles_localised(
            key,
            n=n_examples,
        )

        mf_position_errors = []
        anchor_position_errors = []
        mf_presence_probs = []
        true_presence_all = []

        for example_idx in range(n_examples):
            mf_presence, mf_position, _ = matched_filter_slot_estimates(
                images_eval[example_idx],
            )

            present_mask = theta_eval["presence"][example_idx].astype(bool)

            if present_mask.sum() > 0:
                mf_position_errors.append(
                    jnp.linalg.norm(
                        mf_position[present_mask] - true_positions_eval[example_idx][present_mask],
                        axis=-1,
                    )
                )
                anchor_position_errors.append(
                    jnp.linalg.norm(
                        SLOT_ANCHORS[present_mask] - true_positions_eval[example_idx][present_mask],
                        axis=-1,
                    )
                )

            mf_presence_probs.append(mf_presence)
            true_presence_all.append(theta_eval["presence"][example_idx])

        mf_position_errors = jnp.concatenate(mf_position_errors)
        anchor_position_errors = jnp.concatenate(anchor_position_errors)

        mf_presence_probs = jnp.concatenate(mf_presence_probs)
        true_presence_all = jnp.concatenate(true_presence_all).astype(jnp.float32)

        print("Matched-filter diagnostic")
        print(f"position MAE, matched filter: {mf_position_errors.mean():.2f}px")
        print(f"position MAE, anchors:        {anchor_position_errors.mean():.2f}px")
        print(f"E[score | true present]:      {mf_presence_probs[true_presence_all > 0.5].mean():.3f}")
        print(f"E[score | true absent]:       {mf_presence_probs[true_presence_all < 0.5].mean():.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        axes[0].hist(np.asarray(anchor_position_errors), bins=30, alpha=0.6, label="anchor")
        axes[0].hist(np.asarray(mf_position_errors), bins=30, alpha=0.6, label="matched filter")
        axes[0].set_title("Position error on true-present slots")
        axes[0].set_xlabel("error / px")
        axes[0].legend()

        axes[1].hist(
            np.asarray(mf_presence_probs[true_presence_all < 0.5]),
            bins=30,
            alpha=0.6,
            label="true absent",
        )
        axes[1].hist(
            np.asarray(mf_presence_probs[true_presence_all > 0.5]),
            bins=30,
            alpha=0.6,
            label="true present",
        )
        axes[1].set_title("Matched-filter presence score")
        axes[1].legend()

        example_image = images_eval[0]
        example_presence, example_position, example_residual = matched_filter_slot_estimates(example_image)

        axes[2].imshow(np.asarray(example_residual))
        axes[2].scatter(
            np.asarray(example_position[:, 1]),
            np.asarray(example_position[:, 0]),
            s=20 + 80 * np.asarray(example_presence),
            marker="x",
        )
        axes[2].set_xlim(0, cfg.tile_w)
        axes[2].set_ylim(cfg.tile_h, 0)
        axes[2].set_title("Example residual + estimates")
        axes[2].axis("off")

        fig.tight_layout()
        return fig


    fig_matched_filter_baseline = diagnose_matched_filter_baseline()
    fig_matched_filter_baseline
    return


if __name__ == "__main__":
    app.run()
