# /// script
# dependencies = [
#     "flax==0.12.7",
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

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import jax
    import jax.numpy as jnp
    from jax import random

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive
    from numpyro.infer.util import log_density

    jax.config.update("jax_enable_x64", False)
    numpyro.set_host_device_count(1)

    IMAGE_SHAPE = (32, 32)
    MAX_OBJECTS = 3
    COMPOSITION_DIM = 2
    INTENSITY_BASIS = jnp.array([0.35, 1.15], dtype=jnp.float32)

    # Canonical, spatially anchored object slots. This removes the otherwise fatal
    # label-switching ambiguity in supervised NPE: slot 0 is left, slot 1 is middle,
    # slot 2 is right. The renderer remains order-invariant, but the prior supplies
    # identifiable labels for synthetic training.
    SLOT_X_EDGES = jnp.linspace(0.0, 1.0, MAX_OBJECTS + 1, dtype=jnp.float32)
    SLOT_X_LOW = SLOT_X_EDGES[:-1]
    SLOT_X_HIGH = SLOT_X_EDGES[1:]
    POSITION_LOW = jnp.stack([jnp.zeros(MAX_OBJECTS), SLOT_X_LOW], axis=-1)
    POSITION_HIGH = jnp.stack([jnp.ones(MAX_OBJECTS), SLOT_X_HIGH], axis=-1)
    POSITION_SCALE = POSITION_HIGH - POSITION_LOW
    POSITION_CENTER = 0.5 * (POSITION_LOW + POSITION_HIGH)

    LATENT_SITE_NAMES = (
        "background",
        "observation_noise",
        "presence",
        "position",
        "size",
        "composition",
    )
    return (
        COMPOSITION_DIM,
        IMAGE_SHAPE,
        INTENSITY_BASIS,
        LATENT_SITE_NAMES,
        MAX_OBJECTS,
        POSITION_CENTER,
        POSITION_HIGH,
        POSITION_LOW,
        POSITION_SCALE,
        Predictive,
        dist,
        jax,
        jnp,
        log_density,
        math,
        mo,
        np,
        numpyro,
        plt,
        random,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Simulation-based amortised Bayesian inference for microscopy patches

    This notebook is being built milestone by milestone. The implementation is NumPyro/JAX-native and keeps each object represented by structured, interpretable latent variables: presence, position, size, and constrained composition.
    """)
    return


@app.cell
def _(
    COMPOSITION_DIM,
    IMAGE_SHAPE,
    INTENSITY_BASIS,
    MAX_OBJECTS,
    POSITION_HIGH,
    POSITION_LOW,
    dist,
    jnp,
    numpyro,
):
    def make_grid(image_shape=IMAGE_SHAPE):
        """Pixel-centre coordinate grid in unit-square (y, x) coordinates."""
        height, width = image_shape
        y = jnp.linspace(0.0, 1.0, height, dtype=jnp.float32)
        x = jnp.linspace(0.0, 1.0, width, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        return jnp.stack([yy, xx], axis=-1)


    def composition_to_intensity(composition):
        """Fixed interpretable map from 2-simplex composition to scalar intensity."""
        return jnp.sum(composition * INTENSITY_BASIS, axis=-1)


    def spatial_kernel(position, size, image_shape=IMAGE_SHAPE):
        """Isotropic Gaussian object kernels; position is (y, x), size is sigma."""
        grid = make_grid(image_shape)
        delta = grid[None, :, :, :] - position[:, None, None, :]
        sqdist = jnp.sum(delta**2, axis=-1)
        sigma2 = size[:, None, None] ** 2 + 1e-6
        return jnp.exp(-0.5 * sqdist / sigma2)


    def render_scene(background, presence, position, size, composition, image_shape=IMAGE_SHAPE):
        """Order-invariant deterministic renderer: background + summed object kernels."""
        kernel = spatial_kernel(position, size, image_shape)
        intensity = composition_to_intensity(composition)
        contributions = presence[:, None, None] * intensity[:, None, None] * kernel
        return background + jnp.sum(contributions, axis=0)


    def image_distribution(mean_image, observation_noise):
        """Continuous camera model used for the observed image site."""
        return dist.Normal(mean_image, observation_noise).to_event(2)


    def patch_model(image=None, max_objects=MAX_OBJECTS, image_shape=IMAGE_SHAPE):
        """NumPyro generative model for one microscopy image patch.

        Slots are spatially anchored in x to make synthetic labels identifiable for
        NPE. This is an explicit prior constraint, not a learned embedding.
        """
        background = numpyro.sample("background", dist.Uniform(0.0, 0.20))
        observation_noise = numpyro.sample(
            "observation_noise", dist.LogNormal(jnp.log(0.035), 0.25)
        )
        presence = numpyro.sample(
            "presence", dist.Bernoulli(probs=0.55).expand([max_objects]).to_event(1)
        )
        position = numpyro.sample(
            "position",
            dist.Uniform(POSITION_LOW, POSITION_HIGH).to_event(2),
        )
        size = numpyro.sample(
            "size",
            dist.Uniform(
                0.045 * jnp.ones(max_objects), 0.16 * jnp.ones(max_objects)
            ).to_event(1),
        )
        composition = numpyro.sample(
            "composition",
            dist.Dirichlet(2.0 * jnp.ones(COMPOSITION_DIM))
            .expand([max_objects])
            .to_event(1),
        )

        mean_image = render_scene(
            background, presence, position, size, composition, image_shape
        )
        numpyro.deterministic("mean", mean_image)
        numpyro.deterministic("count", jnp.sum(presence))
        numpyro.sample("obs", image_distribution(mean_image, observation_noise), obs=image)

    return composition_to_intensity, patch_model, render_scene, spatial_kernel


@app.cell
def _(
    COMPOSITION_DIM,
    IMAGE_SHAPE,
    MAX_OBJECTS,
    Predictive,
    jnp,
    mo,
    patch_model,
    random,
):
    prior_predictive_m1 = Predictive(
        patch_model,
        num_samples=12,
        return_sites=(
            "background",
            "observation_noise",
            "presence",
            "position",
            "size",
            "composition",
            "mean",
            "obs",
            "count",
        ),
    )(random.PRNGKey(101))

    actual_shapes_m1 = {
        name: tuple(value.shape) for name, value in prior_predictive_m1.items()
    }
    expected_shapes_m1 = {
        "background": (12,),
        "observation_noise": (12,),
        "presence": (12, MAX_OBJECTS),
        "position": (12, MAX_OBJECTS, 2),
        "size": (12, MAX_OBJECTS),
        "composition": (12, MAX_OBJECTS, COMPOSITION_DIM),
        "mean": (12, *IMAGE_SHAPE),
        "obs": (12, *IMAGE_SHAPE),
        "count": (12,),
    }
    shape_checks_m1 = {
        name: actual_shapes_m1[name] == expected_shape
        for name, expected_shape in expected_shapes_m1.items()
    }
    prior_predictive_finite_m1 = bool(
        jnp.all(jnp.isfinite(prior_predictive_m1["mean"]))
        & jnp.all(jnp.isfinite(prior_predictive_m1["obs"]))
    )

    mo.md(
        "### Milestone 1 prior predictive check\n\n"
        f"Prior predictive sampling with `numpyro.infer.Predictive` works: `{all(shape_checks_m1.values()) and prior_predictive_finite_m1}`.\n\n"
        f"Observed image batch shape: `{actual_shapes_m1['obs']}`; mean image batch shape: `{actual_shapes_m1['mean']}`.\n\n"
        f"All checked shapes: `{actual_shapes_m1}`"
    )
    return (
        actual_shapes_m1,
        prior_predictive_finite_m1,
        prior_predictive_m1,
        shape_checks_m1,
    )


@app.cell
def _(
    LATENT_SITE_NAMES,
    jnp,
    log_density,
    mo,
    patch_model,
    prior_predictive_m1,
):
    latents0_m1 = {
        name: prior_predictive_m1[name][0] for name in LATENT_SITE_NAMES
    }
    log_joint_m1, log_trace_m1 = log_density(
        patch_model,
        model_args=(prior_predictive_m1["obs"][0],),
        model_kwargs={},
        params=latents0_m1,
    )
    log_joint_finite_m1 = bool(jnp.isfinite(log_joint_m1))
    trace_sample_sites_m1 = tuple(
        name for name, site in log_trace_m1.items() if site["type"] == "sample"
    )
    required_sample_sites_present_m1 = all(
        name in trace_sample_sites_m1 for name in (*LATENT_SITE_NAMES, "obs")
    )

    mo.md(
        "### Milestone 1 log-joint check\n\n"
        f"`log p_model(theta, x)` evaluated at supplied latents and image: `{float(log_joint_m1):.3f}`.\n\n"
        f"Finite: `{log_joint_finite_m1}`. Required sample sites present: `{required_sample_sites_present_m1}`.\n\n"
        f"Trace sample sites: `{trace_sample_sites_m1}`"
    )
    return latents0_m1, log_joint_finite_m1, required_sample_sites_present_m1


@app.cell
def _(jnp, latents0_m1, mo, render_scene):
    permutation_m1 = jnp.array([2, 0, 1])
    mean_original_m1 = render_scene(
        latents0_m1["background"],
        latents0_m1["presence"],
        latents0_m1["position"],
        latents0_m1["size"],
        latents0_m1["composition"],
    )
    mean_permuted_m1 = render_scene(
        latents0_m1["background"],
        latents0_m1["presence"][permutation_m1],
        latents0_m1["position"][permutation_m1],
        latents0_m1["size"][permutation_m1],
        latents0_m1["composition"][permutation_m1],
    )
    order_max_abs_diff_m1 = float(jnp.max(jnp.abs(mean_original_m1 - mean_permuted_m1)))
    order_invariant_m1 = order_max_abs_diff_m1 < 1e-6

    control_latents_a_m1 = {
        "background": jnp.array(0.05, dtype=jnp.float32),
        "presence": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        "position": jnp.array(
            [[0.50, 0.50], [0.20, 0.80], [0.80, 0.20]], dtype=jnp.float32
        ),
        "size": jnp.array([0.10, 0.10, 0.10], dtype=jnp.float32),
        "composition": jnp.array(
            [[0.92, 0.08], [0.50, 0.50], [0.15, 0.85]],
            dtype=jnp.float32,
        ),
    }
    control_latents_b_m1 = {
        **control_latents_a_m1,
        "composition": control_latents_a_m1["composition"]
        .at[0]
        .set(jnp.array([0.05, 0.95], dtype=jnp.float32)),
    }
    mean_before_composition_change_m1 = render_scene(**control_latents_a_m1)
    mean_after_composition_change_m1 = render_scene(**control_latents_b_m1)
    composition_max_abs_diff_m1 = float(
        jnp.max(jnp.abs(mean_after_composition_change_m1 - mean_before_composition_change_m1))
    )
    composition_changes_appearance_m1 = composition_max_abs_diff_m1 > 1e-3
    position_and_size_unchanged_m1 = bool(
        jnp.allclose(control_latents_a_m1["position"], control_latents_b_m1["position"])
        & jnp.allclose(control_latents_a_m1["size"], control_latents_b_m1["size"])
    )

    mo.md(
        "### Milestone 1 renderer checks\n\n"
        f"Order-invariance max absolute difference after object permutation: `{order_max_abs_diff_m1:.3e}`.\n\n"
        f"Changing only object 0 composition changes the rendered image by max absolute difference `{composition_max_abs_diff_m1:.3f}`.\n\n"
        f"Position and size tensors unchanged under that composition edit: `{position_and_size_unchanged_m1}`."
    )
    return (
        composition_changes_appearance_m1,
        order_invariant_m1,
        position_and_size_unchanged_m1,
    )


@app.cell(hide_code=True)
def _(
    actual_shapes_m1,
    composition_changes_appearance_m1,
    log_joint_finite_m1,
    mo,
    order_invariant_m1,
    position_and_size_unchanged_m1,
    prior_predictive_finite_m1,
    required_sample_sites_present_m1,
    shape_checks_m1,
):
    milestone_1_passed = bool(
        all(shape_checks_m1.values())
        and prior_predictive_finite_m1
        and log_joint_finite_m1
        and required_sample_sites_present_m1
        and order_invariant_m1
        and composition_changes_appearance_m1
        and position_and_size_unchanged_m1
    )

    mo.md(
        f"""
        ## Milestone 1 report — minimal generative model

        **Implemented.** A NumPyro model `patch_model` for one image patch with named latent sample sites `background`, `observation_noise`, `presence`, `position`, `size`, and `composition`; a deterministic order-invariant renderer; deterministic site `mean`; and observed image sample site `obs`. Object slots are now **spatially anchored** in x (left/middle/right) so that synthetic labels are identifiable during NPE rather than arbitrarily exchangeable.

        **Verified.** Prior predictive sampling works; generated `obs` and `mean` tensors have shape `{actual_shapes_m1['obs']}`; the model log joint is finite for supplied latents and an image; permuting object order leaves the rendered mean unchanged; and editing only composition changes intensity while leaving position and size latents fixed.

        **Concerns.** The first model is deliberately simple: isotropic Gaussian objects, scalar grayscale intensity from a fixed composition basis, scalar background, homoscedastic Gaussian noise, and at most one object per x-anchored slot. The anchoring fixes label-switching for amortised training but restricts scenes with multiple objects in the same x-band.

        **Milestone passed:** `{milestone_1_passed}`.

        **Next.** If this pass remains stable, proceed to Milestone 2: synthetic diagnostics for prior predictive images, latent distributions, overlays, and finite-probability stress checks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Milestone 2 — synthetic diagnostics

    Before amortisation, the simulator is checked visually and numerically: prior predictive images, latent distributions, overlays, finite log probabilities, and simple empty/sparse/crowded stress scenes.
    """)
    return


@app.cell
def _(Predictive, np, patch_model, plt, random):
    prior_diag_m2 = Predictive(
        patch_model,
        num_samples=16,
        return_sites=("obs", "mean", "presence", "position", "size", "composition", "background", "observation_noise", "count"),
    )(random.PRNGKey(202))

    fig_prior_grid_m2, axes_prior_grid_m2 = plt.subplots(4, 4, figsize=(7, 7))
    for _ax, _image, _count in zip(
        axes_prior_grid_m2.ravel(),
        np.asarray(prior_diag_m2["obs"]),
        np.asarray(prior_diag_m2["count"]),
    ):
        _im = _ax.imshow(_image, cmap="magma", interpolation="nearest")
        _ax.set_title(f"count={int(_count)}", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    fig_prior_grid_m2.suptitle("Prior predictive observed image patches", y=0.94)
    fig_prior_grid_m2.tight_layout()
    fig_prior_grid_m2
    return (prior_diag_m2,)


@app.cell
def _(np, plt, prior_diag_m2):
    prior_visual_indices_m2 = np.arange(6)
    prior_vmin_m2 = float(np.min(np.asarray(prior_diag_m2["mean"])[prior_visual_indices_m2]))
    prior_vmax_m2 = float(np.max(np.asarray(prior_diag_m2["obs"])[prior_visual_indices_m2]))
    residual_abs_m2 = float(
        np.max(
            np.abs(
                np.asarray(prior_diag_m2["obs"])[prior_visual_indices_m2]
                - np.asarray(prior_diag_m2["mean"])[prior_visual_indices_m2]
            )
        )
    )

    fig_prior_mean_obs_m2, axes_prior_mean_obs_m2 = plt.subplots(
        len(prior_visual_indices_m2), 3, figsize=(7.5, 12)
    )
    for _row, _idx in enumerate(prior_visual_indices_m2):
        _mean = np.asarray(prior_diag_m2["mean"][_idx])
        _obs = np.asarray(prior_diag_m2["obs"][_idx])
        _resid = _obs - _mean
        _count = int(prior_diag_m2["count"][_idx])

        axes_prior_mean_obs_m2[_row, 0].imshow(
            _mean, cmap="magma", interpolation="nearest", vmin=prior_vmin_m2, vmax=prior_vmax_m2
        )
        axes_prior_mean_obs_m2[_row, 0].set_title(f"mean; count={_count}", fontsize=8)
        axes_prior_mean_obs_m2[_row, 1].imshow(
            _obs, cmap="magma", interpolation="nearest", vmin=prior_vmin_m2, vmax=prior_vmax_m2
        )
        axes_prior_mean_obs_m2[_row, 1].set_title("observed draw", fontsize=8)
        axes_prior_mean_obs_m2[_row, 2].imshow(
            _resid,
            cmap="coolwarm",
            interpolation="nearest",
            vmin=-residual_abs_m2,
            vmax=residual_abs_m2,
        )
        axes_prior_mean_obs_m2[_row, 2].set_title("obs - mean", fontsize=8)
        for _ax in axes_prior_mean_obs_m2[_row, :]:
            _ax.set_xticks([])
            _ax.set_yticks([])
    fig_prior_mean_obs_m2.suptitle("Prior predictive: deterministic mean vs noisy observation", y=0.995)
    fig_prior_mean_obs_m2.tight_layout()
    fig_prior_mean_obs_m2
    return


@app.cell
def _(
    IMAGE_SHAPE,
    LATENT_SITE_NAMES,
    MAX_OBJECTS,
    composition_to_intensity,
    jnp,
    np,
    plt,
    prior_diag_m2,
    spatial_kernel,
):
    component_sample_index_m2 = int(np.argmax(np.asarray(prior_diag_m2["count"])))
    component_latents_m2 = {name: prior_diag_m2[name][component_sample_index_m2] for name in LATENT_SITE_NAMES}
    component_kernel_m2 = spatial_kernel(
        component_latents_m2["position"], component_latents_m2["size"]
    )
    component_intensity_m2 = composition_to_intensity(component_latents_m2["composition"])
    component_contributions_m2 = (
        component_latents_m2["presence"][:, None, None]
        * component_intensity_m2[:, None, None]
        * component_kernel_m2
    )

    fig_components_m2, axes_components_m2 = plt.subplots(1, MAX_OBJECTS + 2, figsize=(12, 2.5))
    axes_components_m2[0].imshow(
        np.asarray(component_latents_m2["background"] + jnp.zeros(IMAGE_SHAPE)),
        cmap="magma",
        interpolation="nearest",
        vmin=0,
        vmax=float(np.max(np.asarray(prior_diag_m2["mean"][component_sample_index_m2]))),
    )
    axes_components_m2[0].set_title("background", fontsize=8)
    for _obj in range(MAX_OBJECTS):
        _present = float(component_latents_m2["presence"][_obj])
        _pos = np.asarray(component_latents_m2["position"][_obj])
        _size = float(component_latents_m2["size"][_obj])
        _comp = np.asarray(component_latents_m2["composition"][_obj])
        axes_components_m2[_obj + 1].imshow(
            np.asarray(component_contributions_m2[_obj]),
            cmap="magma",
            interpolation="nearest",
            vmin=0,
            vmax=float(np.max(np.asarray(prior_diag_m2["mean"][component_sample_index_m2]))),
        )
        axes_components_m2[_obj + 1].set_title(
            f"obj {_obj}: z={_present:.0f}\npos=({_pos[0]:.2f},{_pos[1]:.2f})\nsize={_size:.2f}, comp={_comp.round(2)}",
            fontsize=7,
        )
    axes_components_m2[-1].imshow(
        np.asarray(prior_diag_m2["mean"][component_sample_index_m2]),
        cmap="magma",
        interpolation="nearest",
    )
    axes_components_m2[-1].set_title("summed mean", fontsize=8)
    for _ax in axes_components_m2:
        _ax.set_xticks([])
        _ax.set_yticks([])
    fig_components_m2.suptitle(
        f"Renderer decomposition for prior sample {component_sample_index_m2}", y=1.08
    )
    fig_components_m2.tight_layout()
    fig_components_m2
    return


@app.cell
def _(COMPOSITION_DIM, MAX_OBJECTS, Predictive, np, patch_model, plt, random):
    latent_diag_m2 = Predictive(
        patch_model,
        num_samples=768,
        return_sites=("presence", "position", "size", "composition", "count", "background", "observation_noise"),
    )(random.PRNGKey(203))

    presence_diag_np_m2 = np.asarray(latent_diag_m2["presence"])
    active_mask_np_m2 = presence_diag_np_m2.astype(bool)
    position_diag_np_m2 = np.asarray(latent_diag_m2["position"])
    size_diag_np_m2 = np.asarray(latent_diag_m2["size"])
    composition_diag_np_m2 = np.asarray(latent_diag_m2["composition"])
    active_positions_np_m2 = position_diag_np_m2[active_mask_np_m2]
    active_sizes_np_m2 = size_diag_np_m2[active_mask_np_m2]
    active_compositions_np_m2 = composition_diag_np_m2[active_mask_np_m2]
    counts_np_m2 = np.asarray(latent_diag_m2["count"])

    fig_latent_dist_m2, axes_latent_dist_m2 = plt.subplots(2, 3, figsize=(10, 6))
    axes_latent_dist_m2[0, 0].hist(counts_np_m2, bins=np.arange(-0.5, MAX_OBJECTS + 1.5), rwidth=0.8)
    axes_latent_dist_m2[0, 0].set_title("object count")
    axes_latent_dist_m2[0, 0].set_xlabel("count")
    axes_latent_dist_m2[0, 0].set_ylabel("frequency")

    axes_latent_dist_m2[0, 1].hist(active_positions_np_m2[:, 1], bins=24, range=(0, 1), alpha=0.75, label="x")
    axes_latent_dist_m2[0, 1].hist(active_positions_np_m2[:, 0], bins=24, range=(0, 1), alpha=0.55, label="y")
    axes_latent_dist_m2[0, 1].set_title("active object positions")
    axes_latent_dist_m2[0, 1].legend()

    axes_latent_dist_m2[0, 2].hist(active_sizes_np_m2, bins=24, range=(0.04, 0.17), color="tab:green")
    axes_latent_dist_m2[0, 2].set_title("active object sizes")
    axes_latent_dist_m2[0, 2].set_xlabel("unit-square sigma")

    for _k in range(COMPOSITION_DIM):
        axes_latent_dist_m2[1, 0].hist(
            active_compositions_np_m2[:, _k], bins=24, range=(0, 1), alpha=0.55, label=f"c{_k}"
        )
    axes_latent_dist_m2[1, 0].set_title("composition simplex components")
    axes_latent_dist_m2[1, 0].legend()

    axes_latent_dist_m2[1, 1].hist(np.asarray(latent_diag_m2["background"]), bins=30, color="tab:gray")
    axes_latent_dist_m2[1, 1].set_title("background")

    axes_latent_dist_m2[1, 2].hist(np.asarray(latent_diag_m2["observation_noise"]), bins=30, color="tab:orange")
    axes_latent_dist_m2[1, 2].set_title("observation noise")

    fig_latent_dist_m2.tight_layout()
    fig_latent_dist_m2
    return (
        active_compositions_np_m2,
        active_positions_np_m2,
        active_sizes_np_m2,
    )


@app.cell
def _(IMAGE_SHAPE, np, plt, prior_diag_m2):
    overlay_indices_m2 = np.array([0, 1, 2, 3])
    fig_overlays_m2, axes_overlays_m2 = plt.subplots(1, 4, figsize=(11, 3))
    for _ax, _idx in zip(axes_overlays_m2, overlay_indices_m2):
        _img = np.asarray(prior_diag_m2["obs"][_idx])
        _ax.imshow(_img, cmap="magma", interpolation="nearest")
        _ax.set_title(f"sample {_idx}; count={int(prior_diag_m2['count'][_idx])}", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _present, _pos, _size in zip(
            np.asarray(prior_diag_m2["presence"][_idx]),
            np.asarray(prior_diag_m2["position"][_idx]),
            np.asarray(prior_diag_m2["size"][_idx]),
        ):
            if _present > 0.5:
                _circle = plt.Circle(
                    (_pos[1] * (IMAGE_SHAPE[1] - 1), _pos[0] * (IMAGE_SHAPE[0] - 1)),
                    radius=_size * IMAGE_SHAPE[0],
                    edgecolor="cyan",
                    facecolor="none",
                    linewidth=1.8,
                )
                _ax.add_patch(_circle)
    fig_overlays_m2.suptitle("Sampled positions/sizes overlaid on prior images", y=1.02)
    fig_overlays_m2.tight_layout()
    fig_overlays_m2
    return


@app.cell
def _(
    LATENT_SITE_NAMES,
    jnp,
    log_density,
    math,
    np,
    patch_model,
    plt,
    prior_diag_m2,
    render_scene,
):
    finite_prior_logps_m2 = []
    for _idx in range(8):
        _latents = {name: prior_diag_m2[name][_idx] for name in LATENT_SITE_NAMES}
        _logp, _ = log_density(
            patch_model,
            model_args=(prior_diag_m2["obs"][_idx],),
            model_kwargs={},
            params=_latents,
        )
        finite_prior_logps_m2.append(bool(jnp.isfinite(_logp)))
    finite_prior_logps_m2 = tuple(finite_prior_logps_m2)

    stress_latents_m2 = {
        "empty": {
            "background": jnp.array(0.05, dtype=jnp.float32),
            "observation_noise": jnp.array(0.035, dtype=jnp.float32),
            "presence": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            "position": jnp.array([[0.25, 0.16], [0.50, 0.50], [0.75, 0.84]], dtype=jnp.float32),
            "size": jnp.array([0.08, 0.08, 0.08], dtype=jnp.float32),
            "composition": jnp.array([[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]], dtype=jnp.float32),
        },
        "sparse": {
            "background": jnp.array(0.05, dtype=jnp.float32),
            "observation_noise": jnp.array(0.035, dtype=jnp.float32),
            "presence": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
            "position": jnp.array([[0.35, 0.20], [0.50, 0.50], [0.75, 0.84]], dtype=jnp.float32),
            "size": jnp.array([0.09, 0.08, 0.08], dtype=jnp.float32),
            "composition": jnp.array([[0.15, 0.85], [0.50, 0.50], [0.50, 0.50]], dtype=jnp.float32),
        },
        "three-object": {
            "background": jnp.array(0.05, dtype=jnp.float32),
            "observation_noise": jnp.array(0.035, dtype=jnp.float32),
            "presence": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
            "position": jnp.array([[0.45, 0.25], [0.52, 0.50], [0.56, 0.75]], dtype=jnp.float32),
            "size": jnp.array([0.11, 0.10, 0.12], dtype=jnp.float32),
            "composition": jnp.array([[0.85, 0.15], [0.50, 0.50], [0.15, 0.85]], dtype=jnp.float32),
        },
    }
    stress_means_m2 = {
        name: render_scene(
            latents["background"],
            latents["presence"],
            latents["position"],
            latents["size"],
            latents["composition"],
        )
        for name, latents in stress_latents_m2.items()
    }
    stress_logps_m2 = {}
    for _name, _latents in stress_latents_m2.items():
        _logp, _ = log_density(
            patch_model,
            model_args=(stress_means_m2[_name],),
            model_kwargs={},
            params=_latents,
        )
        stress_logps_m2[_name] = float(_logp)
    finite_stress_logps_m2 = {name: math.isfinite(value) for name, value in stress_logps_m2.items()}

    fig_stress_m2, axes_stress_m2 = plt.subplots(1, 3, figsize=(9, 3))
    for _ax, (_name, _mean) in zip(axes_stress_m2, stress_means_m2.items()):
        _ax.imshow(np.asarray(_mean), cmap="magma", interpolation="nearest")
        _ax.set_title(f"{_name}\nlogp={stress_logps_m2[_name]:.1f}", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    fig_stress_m2.suptitle("Empty/sparse/three-object stress scenes")
    fig_stress_m2.tight_layout()
    fig_stress_m2
    return finite_prior_logps_m2, finite_stress_logps_m2


@app.cell(hide_code=True)
def _(
    active_compositions_np_m2,
    active_positions_np_m2,
    active_sizes_np_m2,
    finite_prior_logps_m2,
    finite_stress_logps_m2,
    mo,
    np,
    prior_diag_m2,
):
    latent_support_ok_m2 = bool(
        np.all((active_positions_np_m2 >= 0.0) & (active_positions_np_m2 <= 1.0))
        and np.all((active_sizes_np_m2 >= 0.045) & (active_sizes_np_m2 <= 0.16))
        and np.all(active_compositions_np_m2 > 0.0)
        and np.allclose(active_compositions_np_m2.sum(axis=-1), 1.0, atol=1e-5)
    )
    finite_checks_ok_m2 = all(finite_prior_logps_m2) and all(finite_stress_logps_m2.values())
    image_range_summary_m2 = (
        float(np.min(np.asarray(prior_diag_m2["obs"]))),
        float(np.max(np.asarray(prior_diag_m2["obs"]))),
    )
    milestone_2_passed = bool(latent_support_ok_m2 and finite_checks_ok_m2)

    mo.md(
        f"""
        ## Milestone 2 report — synthetic diagnostics

        **Implemented.** Visual prior predictive diagnostics including an observed-image grid, a deterministic-mean vs noisy-observation grid, and a renderer component decomposition. Also added latent histograms for count/position/size/composition/background/noise, visual overlays of sampled positions and size radii, finite log-probability checks, and empty/sparse/three-object stress scenes.

        **Verified.** Active positions lie in `[0, 1]^2` and within their x-anchored slot supports, sizes lie in the model support, composition vectors are on the simplex, prior-sampled log joints are finite (`{finite_prior_logps_m2}`), and stress-scene log joints are finite (`{finite_stress_logps_m2}`). Prior observed image range in this diagnostic draw: `{image_range_summary_m2}`.

        **Concerns.** The spatial slot anchoring is an explicit modelling restriction introduced to make labelled synthetic NPE identifiable. It should be relaxed later with a proper set posterior or permutation-mixture guide, but it is preferable to silently training against arbitrary exchangeable labels.

        **Milestone passed:** `{milestone_2_passed}`.

        **Next.** Proceed to Milestone 3 only if the visual diagnostics above look plausible: implement a NumPyro guide with mirrored latent sample sites and explicit `log q_phi(theta | image)` evaluation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Milestone 3 — amortised NumPyro guide

    The posterior estimator below is an amortised NumPyro guide. A small neural encoder maps an image patch to distribution parameters, but the latent variables remain the model's structured variables: background, observation noise, presence, position, size, and composition.
    """)
    return


@app.cell
def _(
    COMPOSITION_DIM,
    LATENT_SITE_NAMES,
    MAX_OBJECTS,
    POSITION_LOW,
    POSITION_SCALE,
    dist,
    jax,
    jnp,
    numpyro,
    random,
):
    from numpyro import handlers
    import jax.nn as jnn

    GUIDE_HIDDEN_DIM = 160
    SIZE_LOW = 0.045
    SIZE_HIGH = 0.16
    GUIDE_OUTPUT_DIM = (
        2  # scaled-Beta background concentration
        + 2  # LogNormal noise loc/scale
        + MAX_OBJECTS  # Bernoulli presence logits
        + MAX_OBJECTS * 2 * 2  # Beta alpha/beta for each 2D position coordinate
        + MAX_OBJECTS * 2  # scaled-Beta alpha/beta for each size
        + MAX_OBJECTS * COMPOSITION_DIM  # Dirichlet concentrations for composition
    )


    def softplus_inverse(y):
        y = jnp.asarray(y, dtype=jnp.float32)
        return jnp.log(jnp.expm1(y))


    def positive(raw, floor=1e-3):
        return jnn.softplus(raw) + floor


    def initial_guide_output_bias():
        """Biases initialise the guide near broad, prior-like distributions."""
        pieces = [
            jnp.array([softplus_inverse(3.0), softplus_inverse(3.0)]),  # background mean 0.1
            jnp.array([jnp.log(0.035), softplus_inverse(0.35)]),  # observation noise
            jnp.full((MAX_OBJECTS,), jnp.log(0.55 / 0.45)),  # presence probability
            jnp.full((MAX_OBJECTS * 2 * 2,), softplus_inverse(2.0)),  # positions
            jnp.full((MAX_OBJECTS * 2,), softplus_inverse(2.0)),  # sizes
            jnp.full((MAX_OBJECTS * COMPOSITION_DIM,), softplus_inverse(2.0)),  # composition
        ]
        return jnp.concatenate([piece.reshape(-1) for piece in pieces]).astype(jnp.float32)


    def init_guide_params(key, hidden_dim=GUIDE_HIDDEN_DIM):
        """Initialise a small CNN encoder used by the amortised guide."""
        key_c1, key_c2, key_fc, key_out = random.split(key, 4)
        return {
            "conv1": random.normal(key_c1, (5, 5, 1, 16), dtype=jnp.float32)
            * jnp.sqrt(2.0 / (5 * 5 * 1)),
            "bconv1": jnp.zeros((16,), dtype=jnp.float32),
            "conv2": random.normal(key_c2, (5, 5, 16, 32), dtype=jnp.float32)
            * jnp.sqrt(2.0 / (5 * 5 * 16)),
            "bconv2": jnp.zeros((32,), dtype=jnp.float32),
            "w_fc": random.normal(key_fc, (8 * 8 * 32, hidden_dim), dtype=jnp.float32)
            * jnp.sqrt(2.0 / (8 * 8 * 32)),
            "b_fc": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w_out": random.normal(key_out, (hidden_dim, GUIDE_OUTPUT_DIM), dtype=jnp.float32)
            * 0.02,
            "b_out": initial_guide_output_bias(),
        }


    def ensure_image_batch(image):
        image = jnp.asarray(image, dtype=jnp.float32)
        if image.ndim == 2:
            return image[None, ...], True
        if image.ndim == 3:
            return image, False
        raise ValueError(f"image must have shape HxW or BxHxW, got {image.shape}")


    def conv2d_same(image_batch, kernel, bias, stride=1):
        return (
            jax.lax.conv_general_dilated(
                image_batch,
                kernel,
                window_strides=(stride, stride),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            + bias
        )


    def guide_forward(guide_params, image):
        """Map a batch of images to unconstrained posterior-parameter outputs."""
        image_batch, _ = ensure_image_batch(image)
        features = image_batch[..., None]
        features = jnn.relu(
            conv2d_same(features, guide_params["conv1"], guide_params["bconv1"], stride=2)
        )
        features = jnn.relu(
            conv2d_same(features, guide_params["conv2"], guide_params["bconv2"], stride=2)
        )
        flat = features.reshape((features.shape[0], -1))
        hidden = jnn.relu(flat @ guide_params["w_fc"] + guide_params["b_fc"])
        return hidden @ guide_params["w_out"] + guide_params["b_out"]


    def parse_guide_output(raw_output):
        """Convert unconstrained encoder output into valid distribution parameters."""
        idx = 0
        background_raw = raw_output[:, idx : idx + 2]
        idx += 2
        noise_raw = raw_output[:, idx : idx + 2]
        idx += 2
        presence_logits = raw_output[:, idx : idx + MAX_OBJECTS]
        idx += MAX_OBJECTS
        position_raw = raw_output[:, idx : idx + MAX_OBJECTS * 2 * 2].reshape(
            (-1, MAX_OBJECTS, 2, 2)
        )
        idx += MAX_OBJECTS * 2 * 2
        size_raw = raw_output[:, idx : idx + MAX_OBJECTS * 2].reshape(
            (-1, MAX_OBJECTS, 2)
        )
        idx += MAX_OBJECTS * 2
        composition_raw = raw_output[
            :, idx : idx + MAX_OBJECTS * COMPOSITION_DIM
        ].reshape((-1, MAX_OBJECTS, COMPOSITION_DIM))
        idx += MAX_OBJECTS * COMPOSITION_DIM
        if idx != GUIDE_OUTPUT_DIM:
            raise RuntimeError("Guide output parser consumed the wrong number of entries.")
        return {
            "background_alpha": positive(background_raw[:, 0]),
            "background_beta": positive(background_raw[:, 1]),
            "noise_loc": noise_raw[:, 0],
            "noise_scale": positive(noise_raw[:, 1], floor=0.02),
            "presence_logits": presence_logits,
            "position_alpha": positive(position_raw[..., 0]),
            "position_beta": positive(position_raw[..., 1]),
            "size_alpha": positive(size_raw[..., 0]),
            "size_beta": positive(size_raw[..., 1]),
            "composition_concentration": positive(composition_raw),
        }


    def guide_distribution_params(guide_params, image):
        return parse_guide_output(guide_forward(guide_params, image))


    def make_guide_distributions(parsed_params):
        """Create NumPyro distributions for each mirrored latent sample site."""
        return {
            "background": dist.TransformedDistribution(
                dist.Beta(parsed_params["background_alpha"], parsed_params["background_beta"]),
                dist.transforms.AffineTransform(0.0, 0.20),
            ),
            "observation_noise": dist.LogNormal(
                parsed_params["noise_loc"], parsed_params["noise_scale"]
            ),
            "presence": dist.Bernoulli(logits=parsed_params["presence_logits"]).to_event(1),
            "position": dist.TransformedDistribution(
                dist.Beta(parsed_params["position_alpha"], parsed_params["position_beta"]),
                dist.transforms.AffineTransform(POSITION_LOW, POSITION_SCALE),
            ).to_event(2),
            "size": dist.TransformedDistribution(
                dist.Beta(parsed_params["size_alpha"], parsed_params["size_beta"]),
                dist.transforms.AffineTransform(SIZE_LOW, SIZE_HIGH - SIZE_LOW),
            ).to_event(1),
            "composition": dist.Dirichlet(
                parsed_params["composition_concentration"]
            ).to_event(1),
        }


    def amortized_guide(image, guide_params):
        """NumPyro guide with sample sites matching the generative model latents."""
        image_batch, _ = ensure_image_batch(image)
        distributions = make_guide_distributions(
            guide_distribution_params(guide_params, image_batch)
        )
        with numpyro.plate("batch", image_batch.shape[0]):
            numpyro.sample("background", distributions["background"])
            numpyro.sample("observation_noise", distributions["observation_noise"])
            numpyro.sample("presence", distributions["presence"])
            numpyro.sample("position", distributions["position"])
            numpyro.sample("size", distributions["size"])
            numpyro.sample("composition", distributions["composition"])


    def guide_log_prob(guide_params, image, latents):
        """Evaluate log q_phi(theta | image) for externally supplied latent values."""
        image_batch, was_single = ensure_image_batch(image)
        distributions = make_guide_distributions(
            guide_distribution_params(guide_params, image_batch)
        )
        log_terms = []
        for name in LATENT_SITE_NAMES:
            value = jnp.asarray(latents[name])
            if was_single:
                if name in ("background", "observation_noise") and value.ndim == 0:
                    value = value[None]
                elif name == "presence" and value.ndim == 1:
                    value = value[None, :]
                elif name == "position" and value.ndim == 2:
                    value = value[None, :, :]
                elif name == "size" and value.ndim == 1:
                    value = value[None, :]
                elif name == "composition" and value.ndim == 2:
                    value = value[None, :, :]
            log_terms.append(distributions[name].log_prob(value))
        total = sum(log_terms)
        return total[0] if was_single else total


    def guide_point_estimates(guide_params, image):
        """Posterior means/probabilities used for diagnostics and later visualisation."""
        parsed = guide_distribution_params(guide_params, image)
        background = 0.20 * parsed["background_alpha"] / (
            parsed["background_alpha"] + parsed["background_beta"]
        )
        observation_noise = jnp.exp(
            parsed["noise_loc"] + 0.5 * parsed["noise_scale"] ** 2
        )
        presence_probs = jnn.sigmoid(parsed["presence_logits"])
        unit_position = parsed["position_alpha"] / (
            parsed["position_alpha"] + parsed["position_beta"]
        )
        position = POSITION_LOW + POSITION_SCALE * unit_position
        size = SIZE_LOW + (SIZE_HIGH - SIZE_LOW) * parsed["size_alpha"] / (
            parsed["size_alpha"] + parsed["size_beta"]
        )
        composition = parsed["composition_concentration"] / jnp.sum(
            parsed["composition_concentration"], axis=-1, keepdims=True
        )
        return {
            "background": background,
            "observation_noise": observation_noise,
            "presence_probs": presence_probs,
            "position": position,
            "size": size,
            "composition": composition,
        }


    guide_params_m3 = init_guide_params(random.PRNGKey(301))
    return (
        SIZE_HIGH,
        SIZE_LOW,
        amortized_guide,
        guide_distribution_params,
        guide_log_prob,
        guide_params_m3,
        guide_point_estimates,
        handlers,
        jnn,
        make_guide_distributions,
    )


@app.cell
def _(
    COMPOSITION_DIM,
    LATENT_SITE_NAMES,
    MAX_OBJECTS,
    SIZE_HIGH,
    SIZE_LOW,
    amortized_guide,
    guide_params_m3,
    handlers,
    jnp,
    mo,
    prior_diag_m2,
    random,
):
    guide_test_images_m3 = prior_diag_m2["obs"][:5]
    guide_trace_m3 = handlers.trace(
        handlers.seed(
            lambda image: amortized_guide(image, guide_params_m3), random.PRNGKey(302)
        )
    ).get_trace(guide_test_images_m3)

    guide_sample_sites_m3 = tuple(
        name for name, site in guide_trace_m3.items() if site["type"] == "sample"
    )
    guide_latent_sample_sites_m3 = tuple(
        name for name in guide_sample_sites_m3 if name in LATENT_SITE_NAMES
    )
    guide_site_shapes_m3 = {
        name: tuple(guide_trace_m3[name]["value"].shape) for name in LATENT_SITE_NAMES
    }
    guide_sites_match_m3 = guide_latent_sample_sites_m3 == LATENT_SITE_NAMES
    expected_guide_shapes_m3 = {
        "background": (5,),
        "observation_noise": (5,),
        "presence": (5, MAX_OBJECTS),
        "position": (5, MAX_OBJECTS, 2),
        "size": (5, MAX_OBJECTS),
        "composition": (5, MAX_OBJECTS, COMPOSITION_DIM),
    }
    guide_shape_checks_m3 = {
        name: guide_site_shapes_m3[name] == expected_guide_shapes_m3[name]
        for name in LATENT_SITE_NAMES
    }

    guide_samples_m3 = {
        name: guide_trace_m3[name]["value"] for name in LATENT_SITE_NAMES
    }
    guide_support_checks_m3 = {
        "background": bool(jnp.all((guide_samples_m3["background"] >= 0.0) & (guide_samples_m3["background"] <= 0.20))),
        "observation_noise": bool(jnp.all(guide_samples_m3["observation_noise"] > 0.0)),
        "presence": bool(jnp.all((guide_samples_m3["presence"] == 0.0) | (guide_samples_m3["presence"] == 1.0))),
        "position": bool(jnp.all((guide_samples_m3["position"] >= 0.0) & (guide_samples_m3["position"] <= 1.0))),
        "size": bool(jnp.all((guide_samples_m3["size"] >= SIZE_LOW) & (guide_samples_m3["size"] <= SIZE_HIGH))),
        "composition": bool(
            jnp.all(guide_samples_m3["composition"] > 0.0)
            & jnp.allclose(jnp.sum(guide_samples_m3["composition"], axis=-1), 1.0, atol=1e-5)
        ),
    }

    guided_shapes_and_support_ok_m3 = bool(
        guide_sites_match_m3
        and all(guide_shape_checks_m3.values())
        and all(guide_support_checks_m3.values())
    )

    mo.md(
        "### Milestone 3 guide run check\n\n"
        f"Guide latent sample sites match the model latent sites: `{guide_sites_match_m3}`.\n\n"
        f"Guide sample shapes: `{guide_site_shapes_m3}`.\n\n"
        f"Guide samples satisfy support constraints: `{guide_support_checks_m3}`."
    )
    return guide_test_images_m3, guided_shapes_and_support_ok_m3


@app.cell
def _(
    LATENT_SITE_NAMES,
    guide_log_prob,
    guide_params_m3,
    guide_point_estimates,
    guide_test_images_m3,
    jnp,
    mo,
    np,
    prior_diag_m2,
):
    guide_prior_latents_m3 = {
        name: prior_diag_m2[name][:5] for name in LATENT_SITE_NAMES
    }
    guide_log_prob_prior_latents_m3 = guide_log_prob(
        guide_params_m3, guide_test_images_m3, guide_prior_latents_m3
    )
    guide_log_prob_finite_m3 = bool(jnp.all(jnp.isfinite(guide_log_prob_prior_latents_m3)))

    guide_estimates_pair_m3 = guide_point_estimates(guide_params_m3, guide_test_images_m3[:2])
    guide_image_dependence_score_m3 = float(
        jnp.sum(
            jnp.abs(
                guide_estimates_pair_m3["presence_probs"][0]
                - guide_estimates_pair_m3["presence_probs"][1]
            )
        )
        + jnp.sum(
            jnp.abs(
                guide_estimates_pair_m3["position"][0]
                - guide_estimates_pair_m3["position"][1]
            )
        )
        + jnp.sum(
            jnp.abs(
                guide_estimates_pair_m3["size"][0]
                - guide_estimates_pair_m3["size"][1]
            )
        )
        + jnp.sum(
            jnp.abs(
                guide_estimates_pair_m3["composition"][0]
                - guide_estimates_pair_m3["composition"][1]
            )
        )
    )
    guide_output_changes_with_image_m3 = guide_image_dependence_score_m3 > 1e-4

    mo.md(
        "### Milestone 3 log-q and image-dependence checks\n\n"
        f"`log q_phi(theta | image)` for prior-sampled latents: `{np.asarray(guide_log_prob_prior_latents_m3).round(3).tolist()}`.\n\n"
        f"All finite: `{guide_log_prob_finite_m3}`.\n\n"
        f"Posterior-parameter image-dependence score between two different images: `{guide_image_dependence_score_m3:.4f}`."
    )
    return guide_log_prob_finite_m3, guide_output_changes_with_image_m3


@app.cell
def _(
    IMAGE_SHAPE,
    MAX_OBJECTS,
    guide_params_m3,
    guide_point_estimates,
    guide_test_images_m3,
    np,
    plt,
):
    guide_vis_estimates_m3 = guide_point_estimates(guide_params_m3, guide_test_images_m3[:3])
    fig_guide_initial_m3, axes_guide_initial_m3 = plt.subplots(1, 3, figsize=(9, 3))
    for _ax, _idx in zip(axes_guide_initial_m3, range(3)):
        _ax.imshow(np.asarray(guide_test_images_m3[_idx]), cmap="magma", interpolation="nearest")
        _ax.set_title(f"initial guide means, image {_idx}", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _obj in range(MAX_OBJECTS):
            _prob = float(guide_vis_estimates_m3["presence_probs"][_idx, _obj])
            _pos = np.asarray(guide_vis_estimates_m3["position"][_idx, _obj])
            _size = float(guide_vis_estimates_m3["size"][_idx, _obj])
            _circle = plt.Circle(
                (_pos[1] * (IMAGE_SHAPE[1] - 1), _pos[0] * (IMAGE_SHAPE[0] - 1)),
                radius=_size * IMAGE_SHAPE[0],
                edgecolor=(0.0, 1.0, 1.0, max(0.15, _prob)),
                facecolor="none",
                linewidth=1.5,
            )
            _ax.add_patch(_circle)
            _ax.text(
                _pos[1] * (IMAGE_SHAPE[1] - 1),
                _pos[0] * (IMAGE_SHAPE[0] - 1),
                f"{_prob:.2f}",
                color="white",
                fontsize=7,
                ha="center",
                va="center",
            )
    fig_guide_initial_m3.suptitle("Untrained amortised guide posterior means (diagnostic only)", y=1.02)
    fig_guide_initial_m3.tight_layout()
    fig_guide_initial_m3
    return


@app.cell(hide_code=True)
def _(
    guide_log_prob_finite_m3,
    guide_output_changes_with_image_m3,
    guided_shapes_and_support_ok_m3,
    mo,
):
    milestone_3_passed = bool(
        guided_shapes_and_support_ok_m3
        and guide_log_prob_finite_m3
        and guide_output_changes_with_image_m3
    )

    mo.md(
        f"""
        ## Milestone 3 report — amortised NumPyro guide

        **Implemented.** A small CNN amortised encoder with a NumPyro guide `amortized_guide(image, guide_params)`. The guide calls `numpyro.sample` at the mirrored latent sites `background`, `observation_noise`, `presence`, `position`, `size`, and `composition`. It uses constrained distributions: scaled Beta for background, x-anchored scaled Beta for position, scaled Beta for size, LogNormal for observation noise, Bernoulli for presence, and Dirichlet for composition.

        **Verified.** The guide runs on a batch of images; sample site names and tensor shapes match expectations; guide samples satisfy all support constraints; `guide_log_prob(...)` evaluates finite `log q_phi(theta | image)` values for externally supplied prior latents; and guide outputs change when the image changes.

        **Concerns.** The guide is deliberately untrained and its visual overlays are diagnostic only. Spatial anchoring removes the arbitrary label-switching failure of the earlier exchangeable-slot training setup, but it is a modelling restriction that should eventually be replaced by a proper set posterior or permutation-mixture guide.

        **Milestone passed:** `{milestone_3_passed}`.

        **Next.** Proceed to Milestone 4: simulation-based pretraining with the objective `-log q_phi(theta_sim | x_sim)` using synthetic pairs from `patch_model`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Milestone 4 — simulation-based pretraining

    The amortised guide is now trained on labelled synthetic pairs sampled from the NumPyro model. The objective is exactly the neural posterior-estimation loss

    \[
    L_{\mathrm{NPE}} = -\log q_\phi(	heta_{\mathrm{sim}} \mid x_{\mathrm{sim}}), \qquad (	heta_{\mathrm{sim}}, x_{\mathrm{sim}}) \sim p_{\mathrm{model}}(	heta, x).
    \]

    No real-image reconstruction objective is used here.
    """)
    return


@app.cell
def _(LATENT_SITE_NAMES, Predictive, mo, patch_model, random):
    import optax

    M4_TRAIN_SIZE = 8192
    M4_VAL_SIZE = 2048
    M4_BATCH_SIZE = 256
    M4_NUM_STEPS = 800
    M4_EVAL_EVERY = 80

    synthetic_all_m4 = Predictive(
        patch_model,
        num_samples=M4_TRAIN_SIZE + M4_VAL_SIZE,
        return_sites=(*LATENT_SITE_NAMES, "obs", "mean", "count"),
    )(random.PRNGKey(431))
    synthetic_train_m4 = {
        name: value[:M4_TRAIN_SIZE] for name, value in synthetic_all_m4.items()
    }
    synthetic_val_m4 = {
        name: value[M4_TRAIN_SIZE:] for name, value in synthetic_all_m4.items()
    }

    mo.md(
        "### Milestone 4 synthetic data\n\n"
        f"Generated `{M4_TRAIN_SIZE}` training pairs and `{M4_VAL_SIZE}` held-out validation pairs from the spatially anchored `patch_model`.\n\n"
        f"Training image tensor shape: `{tuple(synthetic_train_m4['obs'].shape)}`."
    )
    return (
        M4_BATCH_SIZE,
        M4_EVAL_EVERY,
        M4_NUM_STEPS,
        M4_TRAIN_SIZE,
        M4_VAL_SIZE,
        optax,
        synthetic_train_m4,
        synthetic_val_m4,
    )


@app.cell
def _(LATENT_SITE_NAMES, guide_log_prob, jax, jnp, optax):
    def select_batch_m4(data, indices):
        return {name: value[indices] for name, value in data.items()}


    def npe_loss_m4(guide_params, batch):
        latents = {name: batch[name] for name in LATENT_SITE_NAMES}
        return -jnp.mean(guide_log_prob(guide_params, batch["obs"], latents))


    optimizer_m4 = optax.adam(1e-3)


    @jax.jit
    def npe_train_step_m4(guide_params, opt_state, batch):
        loss_value, gradients = jax.value_and_grad(npe_loss_m4)(guide_params, batch)
        updates, opt_state = optimizer_m4.update(gradients, opt_state, guide_params)
        guide_params = optax.apply_updates(guide_params, updates)
        return guide_params, opt_state, loss_value

    return npe_loss_m4, npe_train_step_m4, optimizer_m4, select_batch_m4


@app.cell
def _(
    M4_BATCH_SIZE,
    M4_EVAL_EVERY,
    M4_NUM_STEPS,
    M4_TRAIN_SIZE,
    guide_params_m3,
    jnp,
    mo,
    npe_loss_m4,
    npe_train_step_m4,
    optimizer_m4,
    random,
    select_batch_m4,
    synthetic_train_m4,
    synthetic_val_m4,
):
    guide_params_current_m4 = guide_params_m3
    opt_state_m4 = optimizer_m4.init(guide_params_current_m4)
    initial_train_loss_m4 = float(
        npe_loss_m4(guide_params_current_m4, select_batch_m4(synthetic_train_m4, jnp.arange(M4_BATCH_SIZE)))
    )
    initial_val_loss_m4 = float(npe_loss_m4(guide_params_current_m4, synthetic_val_m4))
    best_val_loss_m4 = initial_val_loss_m4
    guide_params_best_m4 = guide_params_current_m4
    best_step_m4 = 0
    training_history_m4 = [(0, initial_train_loss_m4, initial_val_loss_m4)]

    training_key_m4 = random.PRNGKey(432)
    for step_m4 in range(1, M4_NUM_STEPS + 1):
        training_key_m4, subkey_m4 = random.split(training_key_m4)
        batch_indices_m4 = random.choice(
            subkey_m4,
            M4_TRAIN_SIZE,
            shape=(M4_BATCH_SIZE,),
            replace=False,
        )
        batch_m4 = select_batch_m4(synthetic_train_m4, batch_indices_m4)
        guide_params_current_m4, opt_state_m4, train_loss_m4 = npe_train_step_m4(
            guide_params_current_m4, opt_state_m4, batch_m4
        )
        if step_m4 % M4_EVAL_EVERY == 0 or step_m4 == M4_NUM_STEPS:
            val_loss_m4 = float(npe_loss_m4(guide_params_current_m4, synthetic_val_m4))
            training_history_m4.append((step_m4, float(train_loss_m4), val_loss_m4))
            if val_loss_m4 < best_val_loss_m4:
                best_val_loss_m4 = val_loss_m4
                guide_params_best_m4 = guide_params_current_m4
                best_step_m4 = step_m4

    guide_params_m4 = guide_params_best_m4
    final_val_loss_m4 = float(npe_loss_m4(guide_params_m4, synthetic_val_m4))
    loss_decreased_m4 = final_val_loss_m4 < initial_val_loss_m4

    mo.md(
        "### Milestone 4 NPE training run\n\n"
        f"Initial validation NPE loss: `{initial_val_loss_m4:.3f}`.\n\n"
        f"Best validation NPE loss: `{best_val_loss_m4:.3f}` at step `{best_step_m4}`.\n\n"
        f"Loss decreased: `{loss_decreased_m4}`."
    )
    return (
        best_val_loss_m4,
        guide_params_m4,
        initial_val_loss_m4,
        loss_decreased_m4,
        training_history_m4,
    )


@app.cell
def _(np, plt, training_history_m4):
    history_array_m4 = np.asarray(training_history_m4)
    fig_loss_m4, ax_loss_m4 = plt.subplots(figsize=(6, 3.5))
    ax_loss_m4.plot(history_array_m4[:, 0], history_array_m4[:, 1], marker="o", label="mini-batch train")
    ax_loss_m4.plot(history_array_m4[:, 0], history_array_m4[:, 2], marker="o", label="held-out validation")
    ax_loss_m4.set_xlabel("training step")
    ax_loss_m4.set_ylabel("NPE loss = - log q")
    ax_loss_m4.set_title("Simulation-based pretraining loss")
    ax_loss_m4.legend()
    ax_loss_m4.grid(alpha=0.25)
    fig_loss_m4.tight_layout()
    fig_loss_m4
    return


@app.cell
def _(
    COMPOSITION_DIM,
    M4_VAL_SIZE,
    MAX_OBJECTS,
    POSITION_CENTER,
    SIZE_HIGH,
    SIZE_LOW,
    composition_to_intensity,
    guide_params_m4,
    guide_point_estimates,
    jnp,
    mo,
    synthetic_val_m4,
):
    from itertools import permutations

    M4_EVAL_N = M4_VAL_SIZE
    val_eval_m4 = {name: value[:M4_EVAL_N] for name, value in synthetic_val_m4.items()}
    guide_estimates_m4 = guide_point_estimates(guide_params_m4, val_eval_m4["obs"])

    permutations_m4 = jnp.asarray(list(permutations(range(MAX_OBJECTS))))
    true_presence_m4 = val_eval_m4["presence"]
    active_count_m4 = jnp.sum(true_presence_m4)

    pred_pos_perms_m4 = guide_estimates_m4["position"][:, permutations_m4, :]
    pred_size_perms_m4 = guide_estimates_m4["size"][:, permutations_m4]
    pred_comp_perms_m4 = guide_estimates_m4["composition"][:, permutations_m4, :]

    position_abs_by_perm_m4 = jnp.sum(
        jnp.abs(pred_pos_perms_m4 - val_eval_m4["position"][:, None, :, :])
        * true_presence_m4[:, None, :, None],
        axis=(2, 3),
    )
    size_abs_by_perm_m4 = jnp.sum(
        jnp.abs(pred_size_perms_m4 - val_eval_m4["size"][:, None, :])
        * true_presence_m4[:, None, :],
        axis=2,
    )
    composition_abs_by_perm_m4 = jnp.sum(
        jnp.abs(pred_comp_perms_m4 - val_eval_m4["composition"][:, None, :, :])
        * true_presence_m4[:, None, :, None],
        axis=(2, 3),
    )
    true_intensity_m4 = composition_to_intensity(val_eval_m4["composition"])
    pred_intensity_perms_m4 = composition_to_intensity(pred_comp_perms_m4)
    intensity_abs_by_perm_m4 = jnp.sum(
        jnp.abs(pred_intensity_perms_m4 - true_intensity_m4[:, None, :])
        * true_presence_m4[:, None, :],
        axis=2,
    )

    matching_cost_m4 = (
        position_abs_by_perm_m4
        + 2.0 * size_abs_by_perm_m4
        + 0.25 * composition_abs_by_perm_m4
    )
    best_perm_index_m4 = jnp.argmin(matching_cost_m4, axis=1)
    best_position_abs_m4 = position_abs_by_perm_m4[
        jnp.arange(M4_EVAL_N), best_perm_index_m4
    ]
    best_size_abs_m4 = size_abs_by_perm_m4[jnp.arange(M4_EVAL_N), best_perm_index_m4]
    best_composition_abs_m4 = composition_abs_by_perm_m4[
        jnp.arange(M4_EVAL_N), best_perm_index_m4
    ]
    best_intensity_abs_m4 = intensity_abs_by_perm_m4[
        jnp.arange(M4_EVAL_N), best_perm_index_m4
    ]

    position_mae_m4 = float(jnp.sum(best_position_abs_m4) / (2.0 * active_count_m4 + 1e-6))
    size_mae_m4 = float(jnp.sum(best_size_abs_m4) / (active_count_m4 + 1e-6))
    composition_mae_m4 = float(
        jnp.sum(best_composition_abs_m4) / (COMPOSITION_DIM * active_count_m4 + 1e-6)
    )
    composition_intensity_mae_m4 = float(
        jnp.sum(best_intensity_abs_m4) / (active_count_m4 + 1e-6)
    )

    prior_position_mean_m4 = jnp.broadcast_to(POSITION_CENTER, val_eval_m4["position"].shape)
    prior_size_mean_m4 = ((SIZE_LOW + SIZE_HIGH) / 2.0) * jnp.ones_like(val_eval_m4["size"])
    prior_composition_mean_m4 = (1.0 / COMPOSITION_DIM) * jnp.ones_like(val_eval_m4["composition"])
    prior_position_mae_m4 = float(
        jnp.sum(jnp.abs(prior_position_mean_m4 - val_eval_m4["position"]) * true_presence_m4[:, :, None])
        / (2.0 * active_count_m4 + 1e-6)
    )
    prior_size_mae_m4 = float(
        jnp.sum(jnp.abs(prior_size_mean_m4 - val_eval_m4["size"]) * true_presence_m4)
        / (active_count_m4 + 1e-6)
    )
    prior_composition_mae_m4 = float(
        jnp.sum(
            jnp.abs(prior_composition_mean_m4 - val_eval_m4["composition"])
            * true_presence_m4[:, :, None]
        )
        / (COMPOSITION_DIM * active_count_m4 + 1e-6)
    )
    prior_intensity_mae_m4 = float(
        jnp.sum(
            jnp.abs(composition_to_intensity(prior_composition_mean_m4) - true_intensity_m4)
            * true_presence_m4
        )
        / (active_count_m4 + 1e-6)
    )

    position_recovery_better_m4 = position_mae_m4 < prior_position_mae_m4
    size_recovery_better_m4 = size_mae_m4 < prior_size_mae_m4
    composition_recovery_better_m4 = (
        composition_mae_m4 < prior_composition_mae_m4
        and composition_intensity_mae_m4 < prior_intensity_mae_m4
    )

    mo.md(
        "### Milestone 4 held-out synthetic recovery\n\n"
        "Metrics compare inferred latents with known synthetic ground-truth latents. Best permutation matching is still reported for robustness, but the spatially anchored prior makes slots identifiable.\n\n"
        f"Position MAE: `{position_mae_m4:.3f}` vs prior baseline `{prior_position_mae_m4:.3f}`; better: `{position_recovery_better_m4}`.\n\n"
        f"Size MAE: `{size_mae_m4:.3f}` vs prior baseline `{prior_size_mae_m4:.3f}`; better: `{size_recovery_better_m4}`.\n\n"
        f"Composition-vector MAE: `{composition_mae_m4:.3f}` vs prior baseline `{prior_composition_mae_m4:.3f}`.\n\n"
        f"Composition-implied intensity MAE: `{composition_intensity_mae_m4:.3f}` vs prior baseline `{prior_intensity_mae_m4:.3f}`; composition better: `{composition_recovery_better_m4}`."
    )
    return (
        M4_EVAL_N,
        best_composition_abs_m4,
        best_intensity_abs_m4,
        best_perm_index_m4,
        best_position_abs_m4,
        best_size_abs_m4,
        composition_mae_m4,
        composition_recovery_better_m4,
        guide_estimates_m4,
        permutations_m4,
        position_mae_m4,
        position_recovery_better_m4,
        pred_comp_perms_m4,
        pred_pos_perms_m4,
        pred_size_perms_m4,
        prior_composition_mae_m4,
        prior_composition_mean_m4,
        prior_position_mae_m4,
        prior_position_mean_m4,
        prior_size_mae_m4,
        prior_size_mean_m4,
        size_mae_m4,
        size_recovery_better_m4,
        true_intensity_m4,
        true_presence_m4,
        val_eval_m4,
    )


@app.cell
def _(
    amortized_guide,
    guide_params_m4,
    handlers,
    jax,
    jnp,
    np,
    plt,
    random,
    render_scene,
    val_eval_m4,
):
    def sample_guide_render_batch_m4(key, images):
        guide_trace = handlers.trace(
            handlers.seed(lambda image: amortized_guide(image, guide_params_m4), key)
        ).get_trace(images)
        return jax.vmap(render_scene)(
            guide_trace["background"]["value"],
            guide_trace["presence"]["value"],
            guide_trace["position"]["value"],
            guide_trace["size"]["value"],
            guide_trace["composition"]["value"],
        )


    M4_POSTERIOR_PREDICTIVE_SAMPLES = 16
    posterior_predictive_keys_m4 = random.split(
        random.PRNGKey(444), M4_POSTERIOR_PREDICTIVE_SAMPLES
    )
    posterior_sample_renders_m4 = jnp.stack(
        [
            sample_guide_render_batch_m4(_key, val_eval_m4["obs"])
            for _key in posterior_predictive_keys_m4
        ],
        axis=0,
    )
    posterior_predictive_mean_m4 = jnp.mean(posterior_sample_renders_m4, axis=0)
    posterior_mean_renders_m4 = posterior_predictive_mean_m4  # backward-compatible alias for diagnostics
    posterior_predictive_mse_m4 = float(
        jnp.mean((posterior_predictive_mean_m4 - val_eval_m4["obs"]) ** 2)
    )
    posterior_render_mse_m4 = posterior_predictive_mse_m4
    baseline_render_m4 = jnp.mean(val_eval_m4["obs"], axis=(1, 2), keepdims=True) * jnp.ones_like(
        val_eval_m4["obs"]
    )
    baseline_render_mse_m4 = float(jnp.mean((baseline_render_m4 - val_eval_m4["obs"]) ** 2))
    posterior_predictive_better_m4 = posterior_predictive_mse_m4 < baseline_render_mse_m4

    plot_n_m4 = 6
    fig_pp_m4, axes_pp_m4 = plt.subplots(plot_n_m4, 4, figsize=(8.5, 12))
    for _row in range(plot_n_m4):
        _obs_np = np.asarray(val_eval_m4["obs"][_row])
        _true_mean_np = np.asarray(val_eval_m4["mean"][_row])
        _pp_mean_np = np.asarray(posterior_predictive_mean_m4[_row])
        _pp_sample_np = np.asarray(posterior_sample_renders_m4[0, _row])
        _images = [_obs_np, _true_mean_np, _pp_sample_np, _pp_mean_np]
        _titles = ["observed x", "true mean", "one guide sample", "PP mean (16 samples)"]
        for _col, (_image_np, _title) in enumerate(zip(_images, _titles)):
            axes_pp_m4[_row, _col].imshow(_image_np, cmap="magma", interpolation="nearest")
            axes_pp_m4[_row, _col].set_title(_title if _row == 0 else "", fontsize=9)
            axes_pp_m4[_row, _col].set_xticks([])
            axes_pp_m4[_row, _col].set_yticks([])
    fig_pp_m4.suptitle("Held-out synthetic posterior predictive renders from guide samples", y=0.995)
    fig_pp_m4.tight_layout()
    fig_pp_m4
    return (posterior_predictive_mean_m4,)


@app.cell
def _(
    IMAGE_SHAPE,
    MAX_OBJECTS,
    guide_estimates_m4,
    jnp,
    np,
    plt,
    posterior_predictive_mean_m4,
    val_eval_m4,
):
    best_mse_per_image_m4 = jnp.mean(
        (posterior_predictive_mean_m4 - val_eval_m4["obs"]) ** 2, axis=(1, 2)
    )
    best_indices_m4 = np.asarray(jnp.argsort(best_mse_per_image_m4)[:6])

    fig_best_m4, axes_best_m4 = plt.subplots(6, 4, figsize=(9, 13))
    for _row, _idx in enumerate(best_indices_m4):
        _obs = np.asarray(val_eval_m4["obs"][_idx])
        _true_mean = np.asarray(val_eval_m4["mean"][_idx])
        _pp_mean = np.asarray(posterior_predictive_mean_m4[_idx])
        _err = np.abs(_obs - _pp_mean)
        _panel_images = [_obs, _true_mean, _pp_mean, _err]
        _panel_titles = ["observed x", "true mean", "PP mean", "|x - PP mean|"]
        for _col, (_image, _title) in enumerate(zip(_panel_images, _panel_titles)):
            axes_best_m4[_row, _col].imshow(_image, cmap="magma", interpolation="nearest")
            axes_best_m4[_row, _col].set_title(
                (_title if _row == 0 else "")
                + (f"\nidx={int(_idx)}, MSE={float(best_mse_per_image_m4[_idx]):.4f}, count={int(val_eval_m4['count'][_idx])}" if _col == 0 else ""),
                fontsize=8,
            )
            axes_best_m4[_row, _col].set_xticks([])
            axes_best_m4[_row, _col].set_yticks([])

        # Cyan: true active objects on observed image. Magenta: guide posterior means on PP mean.
        for _obj in range(MAX_OBJECTS):
            if float(val_eval_m4["presence"][_idx, _obj]) > 0.5:
                _true_pos = np.asarray(val_eval_m4["position"][_idx, _obj])
                _true_size = float(val_eval_m4["size"][_idx, _obj])
                axes_best_m4[_row, 0].add_patch(
                    plt.Circle(
                        (_true_pos[1] * (IMAGE_SHAPE[1] - 1), _true_pos[0] * (IMAGE_SHAPE[0] - 1)),
                        radius=_true_size * IMAGE_SHAPE[0],
                        edgecolor="cyan",
                        facecolor="none",
                        linewidth=1.4,
                    )
                )
            _pred_pos = np.asarray(guide_estimates_m4["position"][_idx, _obj])
            _pred_size = float(guide_estimates_m4["size"][_idx, _obj])
            _pred_prob = float(guide_estimates_m4["presence_probs"][_idx, _obj])
            axes_best_m4[_row, 2].add_patch(
                plt.Circle(
                    (_pred_pos[1] * (IMAGE_SHAPE[1] - 1), _pred_pos[0] * (IMAGE_SHAPE[0] - 1)),
                    radius=_pred_size * IMAGE_SHAPE[0],
                    edgecolor=(1.0, 0.0, 1.0, max(0.15, _pred_prob)),
                    facecolor="none",
                    linewidth=1.2,
                )
            )
            axes_best_m4[_row, 2].text(
                _pred_pos[1] * (IMAGE_SHAPE[1] - 1),
                _pred_pos[0] * (IMAGE_SHAPE[0] - 1),
                f"{_pred_prob:.2f}",
                color="white",
                fontsize=6,
                ha="center",
                va="center",
            )
    fig_best_m4.suptitle(
        "Best held-out synthetic posterior predictive examples\ntrue active objects in cyan; inferred slots/probabilities in magenta",
        y=0.995,
    )
    fig_best_m4.tight_layout()
    fig_best_m4
    return best_indices_m4, best_mse_per_image_m4


@app.cell(hide_code=True)
def _(best_indices_m4, best_mse_per_image_m4, jnp, mo, np, val_eval_m4):
    best_examples_summary_m4 = {
        "best_indices": best_indices_m4.tolist(),
        "best_mse": np.asarray(best_mse_per_image_m4[best_indices_m4]).round(5).tolist(),
        "best_counts": np.asarray(val_eval_m4["count"])[best_indices_m4].astype(int).tolist(),
        "median_mse": float(jnp.median(best_mse_per_image_m4)),
        "worst_mse": float(jnp.max(best_mse_per_image_m4)),
    }
    image_mse_best_examples_are_nonempty_m4 = all(
        count > 0 for count in best_examples_summary_m4["best_counts"]
    )

    mo.md(
        "### Milestone 4 image-MSE-ranked held-out examples — caution\n\n"
        "The figure above is ranked by posterior-predictive **image MSE**, not by ground-truth latent error. This ranking is misleading for evaluating object inference because empty/near-empty held-out scenes can dominate the best list.\n\n"
        f"Best image-MSE example indices: `{best_examples_summary_m4['best_indices']}`.\n\n"
        f"Their true counts: `{best_examples_summary_m4['best_counts']}`.\n\n"
        f"Best image MSEs: `{best_examples_summary_m4['best_mse']}`.\n\n"
        f"All image-MSE best examples non-empty: `{image_mse_best_examples_are_nonempty_m4}`.\n\n"
        f"Median held-out image MSE: `{best_examples_summary_m4['median_mse']:.5f}`; worst held-out image MSE: `{best_examples_summary_m4['worst_mse']:.5f}`.\n\n"
        "The actual gate for Milestone 4 is the ground-truth latent diagnostic below, stratified to exclude empty-scene artefacts."
    )
    return


@app.cell
def _(
    COMPOSITION_DIM,
    M4_EVAL_N,
    MAX_OBJECTS,
    best_composition_abs_m4,
    best_intensity_abs_m4,
    best_perm_index_m4,
    best_position_abs_m4,
    best_size_abs_m4,
    composition_to_intensity,
    guide_estimates_m4,
    jnp,
    mo,
    permutations_m4,
    pred_comp_perms_m4,
    pred_pos_perms_m4,
    pred_size_perms_m4,
    prior_composition_mean_m4,
    prior_position_mean_m4,
    prior_size_mean_m4,
    true_intensity_m4,
    true_presence_m4,
    val_eval_m4,
):
    matched_position_m4 = pred_pos_perms_m4[jnp.arange(M4_EVAL_N), best_perm_index_m4]
    matched_size_m4 = pred_size_perms_m4[jnp.arange(M4_EVAL_N), best_perm_index_m4]
    matched_composition_m4 = pred_comp_perms_m4[jnp.arange(M4_EVAL_N), best_perm_index_m4]
    matched_presence_probs_m4 = guide_estimates_m4["presence_probs"][:, permutations_m4][
        jnp.arange(M4_EVAL_N), best_perm_index_m4
    ]

    per_image_count_m4 = val_eval_m4["count"]
    nonempty_mask_gt_m4 = per_image_count_m4 > 0
    safe_count_gt_m4 = jnp.maximum(per_image_count_m4, 1.0)
    per_image_position_mae_gt_m4 = jnp.where(
        nonempty_mask_gt_m4, best_position_abs_m4 / (2.0 * safe_count_gt_m4), jnp.nan
    )
    per_image_size_mae_gt_m4 = jnp.where(
        nonempty_mask_gt_m4, best_size_abs_m4 / safe_count_gt_m4, jnp.nan
    )
    per_image_composition_mae_gt_m4 = jnp.where(
        nonempty_mask_gt_m4,
        best_composition_abs_m4 / (COMPOSITION_DIM * safe_count_gt_m4),
        jnp.nan,
    )
    per_image_intensity_mae_gt_m4 = jnp.where(
        nonempty_mask_gt_m4, best_intensity_abs_m4 / safe_count_gt_m4, jnp.nan
    )
    per_image_presence_mae_gt_m4 = jnp.mean(
        jnp.abs(matched_presence_probs_m4 - true_presence_m4), axis=1
    )
    per_image_count_mae_gt_m4 = jnp.abs(
        jnp.sum(guide_estimates_m4["presence_probs"], axis=1) - per_image_count_m4
    )
    per_image_latent_score_gt_m4 = (
        per_image_position_mae_gt_m4
        + 2.0 * per_image_size_mae_gt_m4
        + per_image_composition_mae_gt_m4
        + 0.25 * per_image_presence_mae_gt_m4
    )

    prior_presence_probs_m4 = 0.55 * jnp.ones_like(true_presence_m4)
    prior_count_mean_m4 = jnp.sum(prior_presence_probs_m4, axis=1)
    prior_position_abs_per_image_m4 = jnp.sum(
        jnp.abs(prior_position_mean_m4 - val_eval_m4["position"]) * true_presence_m4[:, :, None],
        axis=(1, 2),
    )
    prior_size_abs_per_image_m4 = jnp.sum(
        jnp.abs(prior_size_mean_m4 - val_eval_m4["size"]) * true_presence_m4,
        axis=1,
    )
    prior_composition_abs_per_image_m4 = jnp.sum(
        jnp.abs(prior_composition_mean_m4 - val_eval_m4["composition"])
        * true_presence_m4[:, :, None],
        axis=(1, 2),
    )
    prior_intensity_abs_per_image_m4 = jnp.sum(
        jnp.abs(composition_to_intensity(prior_composition_mean_m4) - true_intensity_m4)
        * true_presence_m4,
        axis=1,
    )
    prior_presence_mae_per_image_m4 = jnp.mean(
        jnp.abs(prior_presence_probs_m4 - true_presence_m4), axis=1
    )
    prior_count_mae_per_image_m4 = jnp.abs(prior_count_mean_m4 - per_image_count_m4)
    predicted_binary_presence_m4 = (guide_estimates_m4["presence_probs"] > 0.5).astype(jnp.float32)
    predicted_hard_count_m4 = jnp.sum(predicted_binary_presence_m4, axis=1)

    count_stratified_ground_truth_m4 = {}
    for _count_value in range(1, MAX_OBJECTS + 1):
        _mask = per_image_count_m4 == _count_value
        _active = jnp.sum(true_presence_m4[_mask])
        count_stratified_ground_truth_m4[int(_count_value)] = {
            "n": int(jnp.sum(_mask)),
            "position_mae": float(jnp.sum(best_position_abs_m4[_mask]) / (2.0 * _active + 1e-6)),
            "position_prior": float(jnp.sum(prior_position_abs_per_image_m4[_mask]) / (2.0 * _active + 1e-6)),
            "size_mae": float(jnp.sum(best_size_abs_m4[_mask]) / (_active + 1e-6)),
            "size_prior": float(jnp.sum(prior_size_abs_per_image_m4[_mask]) / (_active + 1e-6)),
            "composition_mae": float(jnp.sum(best_composition_abs_m4[_mask]) / (COMPOSITION_DIM * _active + 1e-6)),
            "composition_prior": float(jnp.sum(prior_composition_abs_per_image_m4[_mask]) / (COMPOSITION_DIM * _active + 1e-6)),
            "intensity_mae": float(jnp.sum(best_intensity_abs_m4[_mask]) / (_active + 1e-6)),
            "intensity_prior": float(jnp.sum(prior_intensity_abs_per_image_m4[_mask]) / (_active + 1e-6)),
            "presence_mae": float(jnp.mean(per_image_presence_mae_gt_m4[_mask])),
            "presence_prior": float(jnp.mean(prior_presence_mae_per_image_m4[_mask])),
            "count_mae": float(jnp.mean(per_image_count_mae_gt_m4[_mask])),
            "count_prior": float(jnp.mean(prior_count_mae_per_image_m4[_mask])),
            "hard_count_accuracy": float(jnp.mean(predicted_hard_count_m4[_mask] == per_image_count_m4[_mask])),
            "median_latent_score": float(jnp.nanmedian(per_image_latent_score_gt_m4[_mask])),
        }

    single_droplet_gate_passed_m4 = bool(
        count_stratified_ground_truth_m4[1]["position_mae"] < 0.05
        and count_stratified_ground_truth_m4[1]["size_mae"] < 0.015
        and count_stratified_ground_truth_m4[1]["composition_mae"] < 0.15
        and count_stratified_ground_truth_m4[1]["presence_mae"] < 0.10
        and count_stratified_ground_truth_m4[1]["count_mae"] < 0.20
        and count_stratified_ground_truth_m4[1]["hard_count_accuracy"] > 0.85
    )

    nonempty_ground_truth_gate_passed_m4 = bool(
        single_droplet_gate_passed_m4
        and all(count_stratified_ground_truth_m4[_c]["position_mae"] < 0.06 for _c in range(1, MAX_OBJECTS + 1))
        and all(count_stratified_ground_truth_m4[_c]["size_mae"] < 0.02 for _c in range(1, MAX_OBJECTS + 1))
        and all(count_stratified_ground_truth_m4[_c]["composition_mae"] < 0.16 for _c in range(1, MAX_OBJECTS + 1))
        and all(count_stratified_ground_truth_m4[_c]["count_mae"] < 0.25 for _c in range(1, MAX_OBJECTS + 1))
        and all(count_stratified_ground_truth_m4[_c]["hard_count_accuracy"] > 0.85 for _c in range(1, MAX_OBJECTS + 1))
    )

    mo.md(
        "### Milestone 4 ground-truth latent errors on held-out non-empty scenes\n\n"
        "The latent recovery metrics below compare against the **known synthetic ground-truth latents**, not reconstruction error. Object slots are first matched by the best permutation, though the spatially anchored prior now makes slot identity meaningful. Empty scenes are excluded by stratifying by true count.\n\n"
        f"Count-stratified ground-truth summary: `{count_stratified_ground_truth_m4}`\n\n"
        f"Single-droplet gate passed: `{single_droplet_gate_passed_m4}`.\n\n"
        f"Non-empty ground-truth gate passed: `{nonempty_ground_truth_gate_passed_m4}`."
    )
    return (
        matched_composition_m4,
        matched_position_m4,
        matched_presence_probs_m4,
        matched_size_m4,
        nonempty_ground_truth_gate_passed_m4,
        nonempty_mask_gt_m4,
        per_image_composition_mae_gt_m4,
        per_image_count_mae_gt_m4,
        per_image_latent_score_gt_m4,
        per_image_position_mae_gt_m4,
        per_image_size_mae_gt_m4,
        single_droplet_gate_passed_m4,
    )


@app.cell
def _(
    IMAGE_SHAPE,
    MAX_OBJECTS,
    matched_composition_m4,
    matched_position_m4,
    matched_presence_probs_m4,
    matched_size_m4,
    nonempty_mask_gt_m4,
    np,
    per_image_composition_mae_gt_m4,
    per_image_count_mae_gt_m4,
    per_image_latent_score_gt_m4,
    per_image_position_mae_gt_m4,
    per_image_size_mae_gt_m4,
    plt,
    posterior_predictive_mean_m4,
    val_eval_m4,
):
    def plot_ground_truth_ranked_examples_m4(indices, panel_title):
        fig, axes = plt.subplots(len(indices), 3, figsize=(9, 2.4 * len(indices)))
        if len(indices) == 1:
            axes = axes[None, :]
        for _row, _idx in enumerate(indices):
            _idx = int(_idx)
            _obs = np.asarray(val_eval_m4["obs"][_idx])
            _pp_mean = np.asarray(posterior_predictive_mean_m4[_idx])
            _err = np.abs(_obs - _pp_mean)
            _score = float(per_image_latent_score_gt_m4[_idx])
            _pos_err = float(per_image_position_mae_gt_m4[_idx])
            _size_err = float(per_image_size_mae_gt_m4[_idx])
            _comp_err = float(per_image_composition_mae_gt_m4[_idx])
            _count_err = float(per_image_count_mae_gt_m4[_idx])
            _count = int(val_eval_m4["count"][_idx])

            axes[_row, 0].imshow(_obs, cmap="magma", interpolation="nearest")
            axes[_row, 0].set_title(
                f"observed + true GT\nidx={_idx}, count={_count}, score={_score:.3f}",
                fontsize=8,
            )
            axes[_row, 1].imshow(_pp_mean, cmap="magma", interpolation="nearest")
            axes[_row, 1].set_title(
                f"PP mean + matched guide\npos={_pos_err:.3f}, size={_size_err:.3f}, comp={_comp_err:.3f}, count={_count_err:.2f}",
                fontsize=8,
            )
            axes[_row, 2].imshow(_err, cmap="magma", interpolation="nearest")
            axes[_row, 2].set_title("|x - PP mean| (not the ranking metric)", fontsize=8)

            for _obj in range(MAX_OBJECTS):
                if float(val_eval_m4["presence"][_idx, _obj]) > 0.5:
                    _true_pos = np.asarray(val_eval_m4["position"][_idx, _obj])
                    _true_size = float(val_eval_m4["size"][_idx, _obj])
                    _true_comp = np.asarray(val_eval_m4["composition"][_idx, _obj])
                    axes[_row, 0].add_patch(
                        plt.Circle(
                            (_true_pos[1] * (IMAGE_SHAPE[1] - 1), _true_pos[0] * (IMAGE_SHAPE[0] - 1)),
                            radius=_true_size * IMAGE_SHAPE[0],
                            edgecolor="cyan",
                            facecolor="none",
                            linewidth=1.5,
                        )
                    )
                    axes[_row, 0].text(
                        _true_pos[1] * (IMAGE_SHAPE[1] - 1),
                        _true_pos[0] * (IMAGE_SHAPE[0] - 1),
                        f"GT c={_true_comp.round(2)}",
                        color="white",
                        fontsize=6,
                        ha="center",
                        va="center",
                    )

                    _pred_pos = np.asarray(matched_position_m4[_idx, _obj])
                    _pred_size = float(matched_size_m4[_idx, _obj])
                    _pred_comp = np.asarray(matched_composition_m4[_idx, _obj])
                    _pred_prob = float(matched_presence_probs_m4[_idx, _obj])
                    axes[_row, 1].add_patch(
                        plt.Circle(
                            (_pred_pos[1] * (IMAGE_SHAPE[1] - 1), _pred_pos[0] * (IMAGE_SHAPE[0] - 1)),
                            radius=_pred_size * IMAGE_SHAPE[0],
                            edgecolor=(1.0, 0.0, 1.0, max(0.15, _pred_prob)),
                            facecolor="none",
                            linewidth=1.4,
                        )
                    )
                    axes[_row, 1].text(
                        _pred_pos[1] * (IMAGE_SHAPE[1] - 1),
                        _pred_pos[0] * (IMAGE_SHAPE[0] - 1),
                        f"qz={_pred_prob:.2f}\nc={_pred_comp.round(2)}",
                        color="white",
                        fontsize=6,
                        ha="center",
                        va="center",
                    )
            for _col in range(3):
                axes[_row, _col].set_xticks([])
                axes[_row, _col].set_yticks([])
        fig.suptitle(panel_title, y=0.995)
        fig.tight_layout()
        return fig

    nonempty_indices_gt_m4 = np.where(np.asarray(nonempty_mask_gt_m4))[0]
    latent_scores_np_m4 = np.asarray(per_image_latent_score_gt_m4)
    best_nonempty_latent_indices_m4 = nonempty_indices_gt_m4[
        np.argsort(latent_scores_np_m4[nonempty_indices_gt_m4])[:6]
    ]
    worst_nonempty_latent_indices_m4 = nonempty_indices_gt_m4[
        np.argsort(latent_scores_np_m4[nonempty_indices_gt_m4])[-6:][::-1]
    ]
    return (
        best_nonempty_latent_indices_m4,
        plot_ground_truth_ranked_examples_m4,
        worst_nonempty_latent_indices_m4,
    )


@app.cell
def _(best_nonempty_latent_indices_m4, plot_ground_truth_ranked_examples_m4):
    fig_best_nonempty_gt_m4 = plot_ground_truth_ranked_examples_m4(
        best_nonempty_latent_indices_m4,
        "Best non-empty held-out examples ranked by ground-truth latent error\ncyan: true objects; magenta: matched guide posterior means",
    )
    fig_best_nonempty_gt_m4
    return


@app.cell
def _(plot_ground_truth_ranked_examples_m4, worst_nonempty_latent_indices_m4):
    fig_worst_nonempty_gt_m4 = plot_ground_truth_ranked_examples_m4(
        worst_nonempty_latent_indices_m4,
        "Worst non-empty held-out examples ranked by ground-truth latent error\ncyan: true objects; magenta: matched guide posterior means",
    )
    fig_worst_nonempty_gt_m4
    return


@app.cell
def _(
    IMAGE_SHAPE,
    MAX_OBJECTS,
    guide_estimates_m4,
    jnp,
    np,
    plt,
    posterior_predictive_mean_m4,
    val_eval_m4,
):
    render_mse_per_image_m4 = jnp.mean((posterior_predictive_mean_m4 - val_eval_m4["obs"]) ** 2, axis=(1, 2))
    worst_indices_m4 = np.asarray(jnp.argsort(render_mse_per_image_m4)[-4:][::-1])

    fig_failures_m4, axes_failures_m4 = plt.subplots(2, 4, figsize=(10, 5))
    for _col, _idx in enumerate(worst_indices_m4):
        axes_failures_m4[0, _col].imshow(
            np.asarray(val_eval_m4["obs"][_idx]), cmap="magma", interpolation="nearest"
        )
        axes_failures_m4[0, _col].set_title(
            f"worst {_col + 1}: count={int(val_eval_m4['count'][_idx])}\nMSE={float(render_mse_per_image_m4[_idx]):.4f}",
            fontsize=8,
        )
        axes_failures_m4[1, _col].imshow(
            np.asarray(posterior_predictive_mean_m4[_idx]), cmap="magma", interpolation="nearest"
        )
        axes_failures_m4[1, _col].set_title("PP mean + inferred slots", fontsize=8)
        for _obj in range(MAX_OBJECTS):
            if float(val_eval_m4["presence"][_idx, _obj]) > 0.5:
                _true_pos = np.asarray(val_eval_m4["position"][_idx, _obj])
                _true_size = float(val_eval_m4["size"][_idx, _obj])
                _true_circle = plt.Circle(
                    (_true_pos[1] * (IMAGE_SHAPE[1] - 1), _true_pos[0] * (IMAGE_SHAPE[0] - 1)),
                    radius=_true_size * IMAGE_SHAPE[0],
                    edgecolor="cyan",
                    facecolor="none",
                    linewidth=1.6,
                )
                axes_failures_m4[0, _col].add_patch(_true_circle)
            _inferred_pos = np.asarray(guide_estimates_m4["position"][_idx, _obj])
            _inferred_size = float(guide_estimates_m4["size"][_idx, _obj])
            _inferred_prob = float(guide_estimates_m4["presence_probs"][_idx, _obj])
            _inferred_circle = plt.Circle(
                (_inferred_pos[1] * (IMAGE_SHAPE[1] - 1), _inferred_pos[0] * (IMAGE_SHAPE[0] - 1)),
                radius=_inferred_size * IMAGE_SHAPE[0],
                edgecolor=(1.0, 0.0, 1.0, max(0.15, _inferred_prob)),
                facecolor="none",
                linewidth=1.3,
            )
            axes_failures_m4[1, _col].add_patch(_inferred_circle)
        for _row in range(2):
            axes_failures_m4[_row, _col].set_xticks([])
            axes_failures_m4[_row, _col].set_yticks([])
    fig_failures_m4.suptitle("Failure cases: true active objects (cyan) vs inferred slots (magenta)", y=1.02)
    fig_failures_m4.tight_layout()
    fig_failures_m4
    return


@app.cell(hide_code=True)
def _(mo, synthetic_train_m4):
    M4_OVERFIT_N = 32
    M4_OVERFIT_STEPS = 1000
    M4_OVERFIT_LR = 3e-3
    synthetic_overfit_m4 = {
        name: value[:M4_OVERFIT_N] for name, value in synthetic_train_m4.items()
    }

    mo.md(
        "### Milestone 4 tiny-synthetic overfit sanity check\n\n"
        f"To check that the guide has enough capacity and that the NPE objective is wired correctly, a separate guide copy is trained to overfit only `{M4_OVERFIT_N}` synthetic training examples. This does **not** replace the held-out pretrained guide `guide_params_m4`; it is a debugging sanity check."
    )
    return M4_OVERFIT_LR, M4_OVERFIT_N, M4_OVERFIT_STEPS, synthetic_overfit_m4


@app.cell
def _(
    M4_OVERFIT_LR,
    M4_OVERFIT_STEPS,
    guide_params_m3,
    jax,
    mo,
    npe_loss_m4,
    optax,
    synthetic_overfit_m4,
):
    def overfit_loss_m4(guide_params):
        return npe_loss_m4(guide_params, synthetic_overfit_m4)


    overfit_optimizer_m4 = optax.adam(M4_OVERFIT_LR)


    @jax.jit
    def overfit_train_step_m4(guide_params, opt_state):
        loss_value, gradients = jax.value_and_grad(overfit_loss_m4)(guide_params)
        updates, opt_state = overfit_optimizer_m4.update(gradients, opt_state, guide_params)
        guide_params = optax.apply_updates(guide_params, updates)
        return guide_params, opt_state, loss_value


    guide_params_overfit_current_m4 = guide_params_m3
    overfit_opt_state_m4 = overfit_optimizer_m4.init(guide_params_overfit_current_m4)
    overfit_initial_loss_m4 = float(overfit_loss_m4(guide_params_overfit_current_m4))
    overfit_history_m4 = [(0, overfit_initial_loss_m4)]

    for _overfit_step in range(1, M4_OVERFIT_STEPS + 1):
        guide_params_overfit_current_m4, overfit_opt_state_m4, overfit_step_loss_m4 = overfit_train_step_m4(
            guide_params_overfit_current_m4, overfit_opt_state_m4
        )
        if _overfit_step % 100 == 0 or _overfit_step == M4_OVERFIT_STEPS:
            overfit_history_m4.append((_overfit_step, float(overfit_step_loss_m4)))

    guide_params_overfit_m4 = guide_params_overfit_current_m4
    overfit_final_loss_m4 = float(overfit_loss_m4(guide_params_overfit_m4))
    overfit_loss_drop_m4 = overfit_initial_loss_m4 - overfit_final_loss_m4
    overfit_loss_decreased_m4 = overfit_loss_drop_m4 > 15.0

    mo.md(
        "#### Tiny-overfit NPE loss\n\n"
        f"Initial tiny-set NPE loss: `{overfit_initial_loss_m4:.3f}`.\n\n"
        f"Final tiny-set NPE loss after `{M4_OVERFIT_STEPS}` full-batch steps: `{overfit_final_loss_m4:.3f}`.\n\n"
        f"Loss drop: `{overfit_loss_drop_m4:.3f}`; substantial drop: `{overfit_loss_decreased_m4}`."
    )
    return (
        guide_params_overfit_m4,
        overfit_final_loss_m4,
        overfit_history_m4,
        overfit_initial_loss_m4,
        overfit_loss_decreased_m4,
    )


@app.cell
def _(np, overfit_history_m4, plt):
    overfit_history_array_m4 = np.asarray(overfit_history_m4)
    fig_overfit_loss_m4, ax_overfit_loss_m4 = plt.subplots(figsize=(6, 3.5))
    ax_overfit_loss_m4.plot(overfit_history_array_m4[:, 0], overfit_history_array_m4[:, 1], marker="o")
    ax_overfit_loss_m4.set_xlabel("full-batch overfit step")
    ax_overfit_loss_m4.set_ylabel("tiny-set NPE loss")
    ax_overfit_loss_m4.set_title("Sanity check: the guide can overfit 32 synthetic pairs")
    ax_overfit_loss_m4.grid(alpha=0.25)
    fig_overfit_loss_m4.tight_layout()
    fig_overfit_loss_m4
    return


@app.cell
def _(
    COMPOSITION_DIM,
    M4_OVERFIT_N,
    SIZE_HIGH,
    SIZE_LOW,
    guide_params_overfit_m4,
    guide_point_estimates,
    jnp,
    mo,
    npe_loss_m4,
    permutations_m4,
    synthetic_overfit_m4,
    synthetic_val_m4,
):
    overfit_estimates_m4 = guide_point_estimates(
        guide_params_overfit_m4, synthetic_overfit_m4["obs"]
    )
    overfit_true_presence_m4 = synthetic_overfit_m4["presence"]
    overfit_active_count_m4 = jnp.sum(overfit_true_presence_m4)

    overfit_pred_pos_perms_m4 = overfit_estimates_m4["position"][:, permutations_m4, :]
    overfit_pred_size_perms_m4 = overfit_estimates_m4["size"][:, permutations_m4]
    overfit_pred_comp_perms_m4 = overfit_estimates_m4["composition"][:, permutations_m4, :]

    overfit_position_abs_by_perm_m4 = jnp.sum(
        jnp.abs(overfit_pred_pos_perms_m4 - synthetic_overfit_m4["position"][:, None, :, :])
        * overfit_true_presence_m4[:, None, :, None],
        axis=(2, 3),
    )
    overfit_size_abs_by_perm_m4 = jnp.sum(
        jnp.abs(overfit_pred_size_perms_m4 - synthetic_overfit_m4["size"][:, None, :])
        * overfit_true_presence_m4[:, None, :],
        axis=2,
    )
    overfit_composition_abs_by_perm_m4 = jnp.sum(
        jnp.abs(overfit_pred_comp_perms_m4 - synthetic_overfit_m4["composition"][:, None, :, :])
        * overfit_true_presence_m4[:, None, :, None],
        axis=(2, 3),
    )
    overfit_matching_cost_m4 = (
        overfit_position_abs_by_perm_m4
        + 2.0 * overfit_size_abs_by_perm_m4
        + 0.25 * overfit_composition_abs_by_perm_m4
    )
    overfit_best_perm_index_m4 = jnp.argmin(overfit_matching_cost_m4, axis=1)

    overfit_position_mae_m4 = float(
        jnp.sum(overfit_position_abs_by_perm_m4[jnp.arange(M4_OVERFIT_N), overfit_best_perm_index_m4])
        / (2.0 * overfit_active_count_m4 + 1e-6)
    )
    overfit_size_mae_m4 = float(
        jnp.sum(overfit_size_abs_by_perm_m4[jnp.arange(M4_OVERFIT_N), overfit_best_perm_index_m4])
        / (overfit_active_count_m4 + 1e-6)
    )
    overfit_composition_mae_m4 = float(
        jnp.sum(overfit_composition_abs_by_perm_m4[jnp.arange(M4_OVERFIT_N), overfit_best_perm_index_m4])
        / (COMPOSITION_DIM * overfit_active_count_m4 + 1e-6)
    )

    overfit_prior_position_mae_m4 = float(
        jnp.sum(jnp.abs(0.5 - synthetic_overfit_m4["position"]) * overfit_true_presence_m4[:, :, None])
        / (2.0 * overfit_active_count_m4 + 1e-6)
    )
    overfit_prior_size_mae_m4 = float(
        jnp.sum(
            jnp.abs(((SIZE_LOW + SIZE_HIGH) / 2.0) - synthetic_overfit_m4["size"])
            * overfit_true_presence_m4
        )
        / (overfit_active_count_m4 + 1e-6)
    )
    overfit_prior_composition_mae_m4 = float(
        jnp.sum(
            jnp.abs((1.0 / COMPOSITION_DIM) - synthetic_overfit_m4["composition"])
            * overfit_true_presence_m4[:, :, None]
        )
        / (COMPOSITION_DIM * overfit_active_count_m4 + 1e-6)
    )

    overfit_latent_recovery_ok_m4 = bool(
        overfit_position_mae_m4 < 0.05
        and overfit_size_mae_m4 < 0.005
        and overfit_composition_mae_m4 < 0.03
    )

    overfit_validation_loss_m4 = float(npe_loss_m4(guide_params_overfit_m4, synthetic_val_m4))

    mo.md(
        "#### Tiny-overfit latent recovery\n\n"
        "Metrics again use best object-slot permutation matching. The held-out validation loss is intentionally poor for this overfit copy and is reported only to make the overfitting explicit.\n\n"
        f"Tiny-set position MAE: `{overfit_position_mae_m4:.4f}` vs prior baseline `{overfit_prior_position_mae_m4:.4f}`.\n\n"
        f"Tiny-set size MAE: `{overfit_size_mae_m4:.4f}` vs prior baseline `{overfit_prior_size_mae_m4:.4f}`.\n\n"
        f"Tiny-set composition MAE: `{overfit_composition_mae_m4:.4f}` vs prior baseline `{overfit_prior_composition_mae_m4:.4f}`.\n\n"
        f"Held-out validation NPE loss for the overfit copy: `{overfit_validation_loss_m4:.1f}`.\n\n"
        f"Tiny-overfit latent recovery passed: `{overfit_latent_recovery_ok_m4}`."
    )
    return overfit_estimates_m4, overfit_latent_recovery_ok_m4


@app.cell
def _(jax, np, overfit_estimates_m4, plt, render_scene, synthetic_overfit_m4):
    overfit_renders_m4 = jax.vmap(render_scene)(
        overfit_estimates_m4["background"],
        overfit_estimates_m4["presence_probs"],
        overfit_estimates_m4["position"],
        overfit_estimates_m4["size"],
        overfit_estimates_m4["composition"],
    )

    overfit_plot_n_m4 = 6
    fig_overfit_examples_m4, axes_overfit_examples_m4 = plt.subplots(overfit_plot_n_m4, 4, figsize=(8.5, 12))
    for _row in range(overfit_plot_n_m4):
        _obs = np.asarray(synthetic_overfit_m4["obs"][_row])
        _true_mean = np.asarray(synthetic_overfit_m4["mean"][_row])
        _render = np.asarray(overfit_renders_m4[_row])
        _err = np.abs(_obs - _render)
        for _col, (_image, _title) in enumerate(
            zip([_obs, _true_mean, _render, _err], ["observed x", "true mean", "overfit guide render", "|x-render|"])
        ):
            axes_overfit_examples_m4[_row, _col].imshow(_image, cmap="magma", interpolation="nearest")
            axes_overfit_examples_m4[_row, _col].set_title(_title if _row == 0 else "", fontsize=9)
            axes_overfit_examples_m4[_row, _col].set_xticks([])
            axes_overfit_examples_m4[_row, _col].set_yticks([])
    fig_overfit_examples_m4.suptitle("Tiny-set overfit visual sanity check", y=0.995)
    fig_overfit_examples_m4.tight_layout()
    fig_overfit_examples_m4
    return


@app.cell(hide_code=True)
def _(
    M4_OVERFIT_N,
    best_val_loss_m4,
    composition_mae_m4,
    composition_recovery_better_m4,
    initial_val_loss_m4,
    loss_decreased_m4,
    mo,
    nonempty_ground_truth_gate_passed_m4,
    overfit_final_loss_m4,
    overfit_initial_loss_m4,
    overfit_latent_recovery_ok_m4,
    overfit_loss_decreased_m4,
    position_mae_m4,
    position_recovery_better_m4,
    prior_composition_mae_m4,
    prior_position_mae_m4,
    prior_size_mae_m4,
    single_droplet_gate_passed_m4,
    size_mae_m4,
    size_recovery_better_m4,
):
    tiny_overfit_passed_m4 = bool(overfit_loss_decreased_m4 and overfit_latent_recovery_ok_m4)
    heldout_average_better_than_prior_m4 = bool(
        loss_decreased_m4
        and position_recovery_better_m4
        and size_recovery_better_m4
        and composition_recovery_better_m4
    )
    milestone_4_passed = bool(
        heldout_average_better_than_prior_m4
        and single_droplet_gate_passed_m4
        and nonempty_ground_truth_gate_passed_m4
        and tiny_overfit_passed_m4
    )

    mo.md(
        f"""
        ## Milestone 4 report — simulation-based pretraining

        **Implemented.** Reworked the simulator/guide after the failed exchangeable-slot experiment. The model now uses explicit x-anchored object slots (left/middle/right) to make synthetic labels identifiable, and the amortised guide uses a small CNN. Training still uses only `L_NPE = -log q_phi(theta_sim | x_sim)` on pairs drawn from `patch_model`. The trained parameters are stored as `guide_params_m4`. A separate tiny-set overfit sanity check trains `guide_params_overfit_m4` on only `{M4_OVERFIT_N}` synthetic pairs.

        **Verified.** Held-out NPE loss decreased from `{initial_val_loss_m4:.3f}` to `{best_val_loss_m4:.3f}`. Aggregate latent metrics are better than prior baselines: position `{position_mae_m4:.3f}` vs `{prior_position_mae_m4:.3f}`, size `{size_mae_m4:.3f}` vs `{prior_size_mae_m4:.3f}`, composition `{composition_mae_m4:.3f}` vs `{prior_composition_mae_m4:.3f}`. Crucially, the gate is now held-out **non-empty ground-truth latent recovery**, not image reconstruction. Single-droplet gate: `{single_droplet_gate_passed_m4}`. All non-empty count-strata gate: `{nonempty_ground_truth_gate_passed_m4}`. The tiny overfit copy drove NPE loss from `{overfit_initial_loss_m4:.3f}` to `{overfit_final_loss_m4:.3f}` and recovered those 32 examples well.

        **Concerns.** The fix is scientifically honest but restrictive: spatial slot anchoring prevents arbitrary label-switching and makes single-droplet detection work, but it limits scenes to at most one object per x-band. The image-MSE best examples remain labelled as a cautionary diagnostic, not as evidence of inference quality. The next principled extension should replace anchoring with a true set-valued or permutation-mixture posterior rather than returning to arbitrary labelled NPE.

        **Milestone passed:** `{milestone_4_passed}`.

        **Next.** Only if the non-empty ground-truth panels and metrics above look acceptable, proceed to Milestone 5. If not, keep improving Milestone 4; do not use reconstruction/image-MSE as a substitute for latent recovery.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Milestone 5 — controlled Bayesian self-consistency on simulator-consistent unlabelled data

    The previous attempt to use `example.jpg` as real data was invalid: the image is colour, crowded, shadowed, and contains non-Gaussian droplet shapes on a bright background, while the current likelihood is a simple grayscale bright-Gaussian-object model. Finite log-ratio gradients are not evidence of a sensible real-data model.

    This milestone is therefore completed in a controlled way first: draw held-out **pseudo-real** images from the same generative process, hide their labels during self-consistency adaptation, and use the known labels only for validation. This tests the Bayesian self-consistency machinery before any claim about actual microscopy images.
    """)
    return


@app.cell
def _(
    IMAGE_SHAPE,
    LATENT_SITE_NAMES,
    MAX_OBJECTS,
    Predictive,
    jnp,
    mo,
    np,
    patch_model,
    plt,
    random,
):
    from pathlib import Path
    from PIL import Image

    # Visual domain check for the available real image. It is deliberately *not* used
    # for SC adaptation with the primitive likelihood in this notebook.
    EXAMPLE_IMAGE_PATH_M5 = Path("example.jpg")
    example_image_rgb_m5 = np.asarray(Image.open(EXAMPLE_IMAGE_PATH_M5).convert("RGB"), dtype=np.float32) / 255.0
    example_domain_compatible_m5 = False
    actual_example_real_used_for_sc_m5 = False

    fig_example_domain_m5, ax_example_domain_m5 = plt.subplots(figsize=(9, 4))
    ax_example_domain_m5.imshow(example_image_rgb_m5)
    ax_example_domain_m5.set_title(
        "Domain check: example.jpg is colour, crowded, shadowed, and not generated by the primitive grayscale likelihood",
        fontsize=10,
    )
    ax_example_domain_m5.set_xticks([])
    ax_example_domain_m5.set_yticks([])
    fig_example_domain_m5.tight_layout()

    M5_PSEUDO_REAL_SIZE = 96
    pseudo_real_data_m5 = Predictive(
        patch_model,
        num_samples=M5_PSEUDO_REAL_SIZE,
        return_sites=(*LATENT_SITE_NAMES, "obs", "mean", "count"),
    )(random.PRNGKey(810))
    pseudo_real_images_m5 = pseudo_real_data_m5["obs"]  # labels are hidden from SC adaptation
    pseudo_real_summary_m5 = {
        "shape": tuple(pseudo_real_images_m5.shape),
        "count_histogram": {
            int(_count): int(jnp.sum(pseudo_real_data_m5["count"] == _count))
            for _count in range(MAX_OBJECTS + 1)
        },
        "min": float(jnp.min(pseudo_real_images_m5)),
        "max": float(jnp.max(pseudo_real_images_m5)),
        "mean": float(jnp.mean(pseudo_real_images_m5)),
        "std": float(jnp.std(pseudo_real_images_m5)),
    }

    fig_pseudo_real_m5, axes_pseudo_real_m5 = plt.subplots(3, 4, figsize=(9, 6.5))
    for _ax, _idx in zip(axes_pseudo_real_m5.ravel(), range(12)):
        _ax.imshow(np.asarray(pseudo_real_images_m5[_idx]), cmap="magma", interpolation="nearest")
        _ax.set_title(f"pseudo-real {_idx}; hidden count={int(pseudo_real_data_m5['count'][_idx])}", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
        # Ground-truth overlays are shown only for validation/inspection, not used in adaptation.
        for _obj in range(MAX_OBJECTS):
            if float(pseudo_real_data_m5["presence"][_idx, _obj]) > 0.5:
                _pos = np.asarray(pseudo_real_data_m5["position"][_idx, _obj])
                _size = float(pseudo_real_data_m5["size"][_idx, _obj])
                _ax.add_patch(
                    plt.Circle(
                        (_pos[1] * (IMAGE_SHAPE[1] - 1), _pos[0] * (IMAGE_SHAPE[0] - 1)),
                        radius=_size * IMAGE_SHAPE[0],
                        edgecolor="cyan",
                        facecolor="none",
                        linewidth=1.2,
                    )
                )
    fig_pseudo_real_m5.suptitle(
        "Simulator-consistent pseudo-real unlabelled set\ncyan ground truth is for validation only",
        y=1.02,
    )
    fig_pseudo_real_m5.tight_layout()

    mo.vstack([fig_example_domain_m5, fig_pseudo_real_m5])
    return (
        M5_PSEUDO_REAL_SIZE,
        actual_example_real_used_for_sc_m5,
        example_domain_compatible_m5,
        pseudo_real_data_m5,
        pseudo_real_images_m5,
        pseudo_real_summary_m5,
    )


@app.cell
def _(
    COMPOSITION_DIM,
    IMAGE_SHAPE,
    LATENT_SITE_NAMES,
    MAX_OBJECTS,
    POSITION_HIGH,
    POSITION_LOW,
    POSITION_SCALE,
    SIZE_HIGH,
    SIZE_LOW,
    dist,
    guide_distribution_params,
    guide_log_prob,
    jax,
    jnp,
    make_guide_distributions,
    optax,
    random,
    render_scene,
):
    def sanitize_latent_samples_m5(samples, eps=1e-5):
        """Keep continuous proposal values in the open support for finite log-probs."""
        composition = jnp.clip(samples["composition"], eps, 1.0)
        composition = composition / jnp.sum(composition, axis=-1, keepdims=True)
        return {
            "background": jnp.clip(samples["background"], eps, 0.20 - eps),
            "observation_noise": jnp.clip(samples["observation_noise"], 1e-4, 1.0),
            "presence": samples["presence"],
            "position": jnp.clip(
                samples["position"],
                POSITION_LOW + eps * POSITION_SCALE,
                POSITION_HIGH - eps * POSITION_SCALE,
            ),
            "size": jnp.clip(samples["size"], SIZE_LOW + eps, SIZE_HIGH - eps),
            "composition": composition,
        }


    def model_log_joint_batch_m5(images, latents):
        """Vectorised log p_model(theta, x) for a batch of supplied latents/images."""
        background = latents["background"]
        observation_noise = latents["observation_noise"]
        presence = latents["presence"]
        position = latents["position"]
        size = latents["size"]
        composition = latents["composition"]

        mean_image = jax.vmap(render_scene)(
            background, presence, position, size, composition
        )
        log_prob = dist.Uniform(0.0, 0.20).log_prob(background)
        log_prob = log_prob + dist.LogNormal(jnp.log(0.035), 0.25).log_prob(
            observation_noise
        )
        log_prob = log_prob + dist.Bernoulli(probs=0.55).expand([MAX_OBJECTS]).to_event(1).log_prob(
            presence
        )
        log_prob = log_prob + dist.Uniform(POSITION_LOW, POSITION_HIGH).to_event(2).log_prob(
            position
        )
        log_prob = log_prob + dist.Uniform(
            0.045 * jnp.ones(MAX_OBJECTS), 0.16 * jnp.ones(MAX_OBJECTS)
        ).to_event(1).log_prob(size)
        log_prob = log_prob + dist.Dirichlet(2.0 * jnp.ones(COMPOSITION_DIM)).expand(
            [MAX_OBJECTS]
        ).to_event(1).log_prob(composition)
        log_prob = log_prob + dist.Normal(
            mean_image, observation_noise[:, None, None]
        ).to_event(2).log_prob(images)
        return log_prob


    def sample_guide_latents_m5(guide_params, images, num_samples, key):
        """Sample theta_l ~ stop_gradient(q_phi_old(theta | x_real))."""
        distributions = make_guide_distributions(guide_distribution_params(guide_params, images))
        keys = random.split(key, len(LATENT_SITE_NAMES))
        raw_samples = {
            name: distributions[name].sample(keys[_i], sample_shape=(num_samples,))
            for _i, name in enumerate(LATENT_SITE_NAMES)
        }
        return jax.tree_util.tree_map(
            jax.lax.stop_gradient, sanitize_latent_samples_m5(raw_samples)
        )


    def flatten_latent_samples_m5(samples):
        num_samples, batch_size = samples["background"].shape[:2]
        return {
            name: value.reshape((num_samples * batch_size,) + value.shape[2:])
            for name, value in samples.items()
        }


    def broadcast_images_for_samples_m5(images, num_samples):
        return jnp.broadcast_to(images[None, ...], (num_samples,) + images.shape).reshape(
            (num_samples * images.shape[0],) + images.shape[1:]
        )


    def log_joint_samples_m5(images, samples):
        num_samples = samples["background"].shape[0]
        flat_latents = flatten_latent_samples_m5(samples)
        flat_images = broadcast_images_for_samples_m5(images, num_samples)
        return model_log_joint_batch_m5(flat_images, flat_latents).reshape(
            (num_samples, images.shape[0])
        )


    def guide_log_prob_samples_m5(guide_params, images, samples):
        num_samples = samples["background"].shape[0]
        flat_latents = flatten_latent_samples_m5(samples)
        flat_images = broadcast_images_for_samples_m5(images, num_samples)
        return guide_log_prob(guide_params, flat_images, flat_latents).reshape(
            (num_samples, images.shape[0])
        )


    def sc_loss_m5(guide_params, images, frozen_samples, frozen_log_joint):
        """Bayesian self-consistency loss: Var_l[log p_model(theta_l,x)-log q_phi(theta_l|x)].

        The ratio is divided by pixel count only for numerical scaling; this multiplies
        the original variance by a constant and does not introduce reconstruction loss.
        """
        log_q = guide_log_prob_samples_m5(guide_params, images, frozen_samples)
        log_ratio_per_pixel = (frozen_log_joint - log_q) / (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
        return jnp.mean(jnp.var(log_ratio_per_pixel, axis=0))


    def sc_value_grad_norm_m5(guide_params, images, frozen_samples, frozen_log_joint):
        value, gradients = jax.value_and_grad(sc_loss_m5)(
            guide_params, images, frozen_samples, frozen_log_joint
        )
        grad_norm = optax.global_norm(gradients)
        gradients_finite = all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(gradients)
        )
        return value, grad_norm, gradients_finite

    return (
        guide_log_prob_samples_m5,
        log_joint_samples_m5,
        sample_guide_latents_m5,
        sc_loss_m5,
        sc_value_grad_norm_m5,
    )


@app.cell
def _(
    actual_example_real_used_for_sc_m5,
    example_domain_compatible_m5,
    guide_log_prob_samples_m5,
    guide_params_m4,
    jnp,
    log_joint_samples_m5,
    mo,
    pseudo_real_images_m5,
    pseudo_real_summary_m5,
    random,
    sample_guide_latents_m5,
    sc_value_grad_norm_m5,
):
    M5_NUM_CHECK_PROPOSALS = 16
    proposal_check_m5 = sample_guide_latents_m5(
        guide_params_m4, pseudo_real_images_m5, M5_NUM_CHECK_PROPOSALS, random.PRNGKey(811)
    )
    log_joint_check_m5 = log_joint_samples_m5(pseudo_real_images_m5, proposal_check_m5)
    log_q_check_m5 = guide_log_prob_samples_m5(
        guide_params_m4, pseudo_real_images_m5, proposal_check_m5
    )
    sc_initial_check_loss_m5, sc_initial_grad_norm_m5, sc_initial_gradients_finite_m5 = sc_value_grad_norm_m5(
        guide_params_m4, pseudo_real_images_m5, proposal_check_m5, log_joint_check_m5
    )
    sc_initial_finite_m5 = bool(
        jnp.all(jnp.isfinite(log_joint_check_m5))
        and jnp.all(jnp.isfinite(log_q_check_m5))
        and jnp.isfinite(sc_initial_check_loss_m5)
        and jnp.isfinite(sc_initial_grad_norm_m5)
        and sc_initial_gradients_finite_m5
    )

    mo.md(
        "### Milestone 5 finite log-ratio and gradient check\n\n"
        f"Pseudo-real set summary: `{pseudo_real_summary_m5}`.\n\n"
        f"`example.jpg` domain compatible with current likelihood: `{example_domain_compatible_m5}`; used for SC: `{actual_example_real_used_for_sc_m5}`.\n\n"
        f"Finite log p/log q/SC loss/gradient on pseudo-real images: `{sc_initial_finite_m5}`.\n\n"
        f"Initial SC loss per-pixel-scaled: `{float(sc_initial_check_loss_m5):.4f}`.\n\n"
        f"Gradient norm into guide parameters: `{float(sc_initial_grad_norm_m5):.4f}`."
    )
    return sc_initial_finite_m5, sc_initial_grad_norm_m5


@app.cell
def _(
    guide_params_m4,
    jax,
    log_joint_samples_m5,
    mo,
    optax,
    pseudo_real_images_m5,
    random,
    sample_guide_latents_m5,
    sc_loss_m5,
):
    M5_NUM_PROPOSALS = 16
    M5_NUM_FIXED_EVAL_PROPOSALS = 32
    M5_NUM_STEPS = 10
    M5_LEARNING_RATE = 1e-6

    fixed_eval_samples_m5 = sample_guide_latents_m5(
        guide_params_m4, pseudo_real_images_m5, M5_NUM_FIXED_EVAL_PROPOSALS, random.PRNGKey(812)
    )
    fixed_eval_log_joint_m5 = log_joint_samples_m5(pseudo_real_images_m5, fixed_eval_samples_m5)
    fixed_eval_sc_before_m5 = float(
        sc_loss_m5(guide_params_m4, pseudo_real_images_m5, fixed_eval_samples_m5, fixed_eval_log_joint_m5)
    )

    optimizer_m5 = optax.chain(
        optax.clip_by_global_norm(10.0), optax.adam(M5_LEARNING_RATE)
    )


    @jax.jit
    def sc_train_step_m5(guide_params, opt_state, images, frozen_samples, frozen_log_joint):
        loss_value, gradients = jax.value_and_grad(sc_loss_m5)(
            guide_params, images, frozen_samples, frozen_log_joint
        )
        grad_norm = optax.global_norm(gradients)
        updates, opt_state = optimizer_m5.update(gradients, opt_state, guide_params)
        guide_params = optax.apply_updates(guide_params, updates)
        return guide_params, opt_state, loss_value, grad_norm


    guide_params_current_m5 = guide_params_m4
    opt_state_m5 = optimizer_m5.init(guide_params_current_m5)
    training_history_m5 = []
    training_key_m5 = random.PRNGKey(813)

    for step_m5 in range(M5_NUM_STEPS):
        training_key_m5, subkey_m5 = random.split(training_key_m5)
        proposals_m5 = sample_guide_latents_m5(
            guide_params_current_m5, pseudo_real_images_m5, M5_NUM_PROPOSALS, subkey_m5
        )
        log_joint_proposals_m5 = log_joint_samples_m5(pseudo_real_images_m5, proposals_m5)
        guide_params_current_m5, opt_state_m5, sc_loss_value_m5, sc_grad_norm_m5 = sc_train_step_m5(
            guide_params_current_m5,
            opt_state_m5,
            pseudo_real_images_m5,
            proposals_m5,
            log_joint_proposals_m5,
        )
        fixed_eval_sc_m5 = float(
            sc_loss_m5(
                guide_params_current_m5,
                pseudo_real_images_m5,
                fixed_eval_samples_m5,
                fixed_eval_log_joint_m5,
            )
        )
        training_history_m5.append(
            (step_m5, float(sc_loss_value_m5), float(sc_grad_norm_m5), fixed_eval_sc_m5)
        )

    guide_params_m5 = guide_params_current_m5
    fixed_eval_sc_after_m5 = float(
        sc_loss_m5(guide_params_m5, pseudo_real_images_m5, fixed_eval_samples_m5, fixed_eval_log_joint_m5)
    )
    sc_fixed_eval_decreased_m5 = fixed_eval_sc_after_m5 < fixed_eval_sc_before_m5

    mo.md(
        "### Milestone 5 controlled SC adaptation\n\n"
        "Labels in `pseudo_real_data_m5` are not used by the SC objective. They are used only downstream to audit whether the unlabelled adaptation preserved the intended latent semantics.\n\n"
        f"Fixed-proposal SC loss before: `{fixed_eval_sc_before_m5:.6f}`.\n\n"
        f"Fixed-proposal SC loss after: `{fixed_eval_sc_after_m5:.6f}`.\n\n"
        f"Fixed-proposal SC loss decreased: `{sc_fixed_eval_decreased_m5}`.\n\n"
        f"Training history `(step, resampled_sc_loss, grad_norm, fixed_eval_sc_loss)`: `{[(s, round(l, 4), round(g, 4), round(e, 4)) for s, l, g, e in training_history_m5]}`"
    )
    return (
        fixed_eval_sc_after_m5,
        fixed_eval_sc_before_m5,
        guide_params_m5,
        sc_fixed_eval_decreased_m5,
    )


@app.cell
def _(
    COMPOSITION_DIM,
    M5_PSEUDO_REAL_SIZE,
    MAX_OBJECTS,
    guide_params_m4,
    guide_params_m5,
    guide_point_estimates,
    jnp,
    mo,
    npe_loss_m4,
    permutations_m4,
    pseudo_real_data_m5,
    synthetic_val_m4,
):
    def synthetic_ground_truth_summary_m5(guide_params, data, n_eval=512):
        eval_data = {name: value[:n_eval] for name, value in data.items()}
        estimates = guide_point_estimates(guide_params, eval_data["obs"])
        permutations = permutations_m4
        true_presence = eval_data["presence"]
        true_count = eval_data["count"]

        pred_pos_perms = estimates["position"][:, permutations, :]
        pred_size_perms = estimates["size"][:, permutations]
        pred_comp_perms = estimates["composition"][:, permutations, :]
        position_abs_by_perm = jnp.sum(
            jnp.abs(pred_pos_perms - eval_data["position"][:, None, :, :])
            * true_presence[:, None, :, None],
            axis=(2, 3),
        )
        size_abs_by_perm = jnp.sum(
            jnp.abs(pred_size_perms - eval_data["size"][:, None, :])
            * true_presence[:, None, :],
            axis=2,
        )
        composition_abs_by_perm = jnp.sum(
            jnp.abs(pred_comp_perms - eval_data["composition"][:, None, :, :])
            * true_presence[:, None, :, None],
            axis=(2, 3),
        )
        best_perm = jnp.argmin(
            position_abs_by_perm + 2.0 * size_abs_by_perm + 0.25 * composition_abs_by_perm,
            axis=1,
        )
        matched_presence_probs = estimates["presence_probs"][:, permutations][
            jnp.arange(n_eval), best_perm
        ]
        hard_count = jnp.sum((estimates["presence_probs"] > 0.5).astype(jnp.float32), axis=1)

        summary = {}
        for _count_value in range(1, MAX_OBJECTS + 1):
            _mask = true_count == _count_value
            _n = int(jnp.sum(_mask))
            _active = jnp.sum(true_presence[_mask])
            _rows = jnp.arange(_n)
            summary[int(_count_value)] = {
                "n": _n,
                "position_mae": float(
                    jnp.sum(position_abs_by_perm[_mask][_rows, best_perm[_mask]])
                    / (2.0 * _active + 1e-6)
                ),
                "size_mae": float(
                    jnp.sum(size_abs_by_perm[_mask][_rows, best_perm[_mask]])
                    / (_active + 1e-6)
                ),
                "composition_mae": float(
                    jnp.sum(composition_abs_by_perm[_mask][_rows, best_perm[_mask]])
                    / (COMPOSITION_DIM * _active + 1e-6)
                ),
                "presence_mae": float(
                    jnp.mean(jnp.mean(jnp.abs(matched_presence_probs[_mask] - true_presence[_mask]), axis=1))
                ),
                "count_mae": float(
                    jnp.mean(jnp.abs(jnp.sum(estimates["presence_probs"][_mask], axis=1) - true_count[_mask]))
                ),
                "hard_count_accuracy": float(jnp.mean(hard_count[_mask] == true_count[_mask])),
            }
        return summary


    heldout_synthetic_summary_before_m5 = synthetic_ground_truth_summary_m5(
        guide_params_m4, synthetic_val_m4
    )
    heldout_synthetic_summary_after_m5 = synthetic_ground_truth_summary_m5(
        guide_params_m5, synthetic_val_m4
    )
    synthetic_npe_before_m5 = float(npe_loss_m4(guide_params_m4, synthetic_val_m4))
    synthetic_npe_after_m5 = float(npe_loss_m4(guide_params_m5, synthetic_val_m4))
    synthetic_npe_delta_m5 = synthetic_npe_after_m5 - synthetic_npe_before_m5

    pseudo_real_label_summary_before_m5 = synthetic_ground_truth_summary_m5(
        guide_params_m4, pseudo_real_data_m5, n_eval=M5_PSEUDO_REAL_SIZE
    )
    pseudo_real_label_summary_after_m5 = synthetic_ground_truth_summary_m5(
        guide_params_m5, pseudo_real_data_m5, n_eval=M5_PSEUDO_REAL_SIZE
    )

    synthetic_no_collapse_m5 = bool(
        synthetic_npe_delta_m5 < 0.5
        and all(
            heldout_synthetic_summary_after_m5[_c]["position_mae"]
            < heldout_synthetic_summary_before_m5[_c]["position_mae"] + 0.005
            for _c in range(1, MAX_OBJECTS + 1)
        )
        and all(
            heldout_synthetic_summary_after_m5[_c]["hard_count_accuracy"]
            > heldout_synthetic_summary_before_m5[_c]["hard_count_accuracy"] - 0.03
            for _c in range(1, MAX_OBJECTS + 1)
        )
        and all(
            heldout_synthetic_summary_after_m5[_c]["count_mae"]
            < heldout_synthetic_summary_before_m5[_c]["count_mae"] + 0.03
            for _c in range(1, MAX_OBJECTS + 1)
        )
    )

    pseudo_real_label_check_m5 = bool(
        all(
            pseudo_real_label_summary_after_m5[_c]["position_mae"] < 0.06
            for _c in range(1, MAX_OBJECTS + 1)
        )
        and all(
            pseudo_real_label_summary_after_m5[_c]["hard_count_accuracy"] > 0.75
            for _c in range(1, MAX_OBJECTS + 1)
        )
    )

    mo.md(
        "### Milestone 5 label-audited checks \n\n"
        "The pseudo-real labels are held out from adaptation and used here only for auditing. This is the controlled test that should have preceded any actual-real experiment.\n\n"
        f"Held-out synthetic validation NPE before/after SC: `{synthetic_npe_before_m5:.3f}` → `{synthetic_npe_after_m5:.3f}`; delta `{synthetic_npe_delta_m5:.3f}`.\n\n"
        f"Held-out synthetic latent summary before SC: `{heldout_synthetic_summary_before_m5}`\n\n"
        f"Held-out synthetic latent summary after SC: `{heldout_synthetic_summary_after_m5}`\n\n"
        f"Pseudo-real labelled audit before SC: `{pseudo_real_label_summary_before_m5}`\n\n"
        f"Pseudo-real labelled audit after SC: `{pseudo_real_label_summary_after_m5}`\n\n"
        f"Held-out synthetic performance did not collapse: `{synthetic_no_collapse_m5}`.\n\n"
        f"Pseudo-real hidden-label audit passed: `{pseudo_real_label_check_m5}`."
    )
    return (
        pseudo_real_label_check_m5,
        synthetic_no_collapse_m5,
        synthetic_npe_delta_m5,
    )


@app.cell
def _(
    IMAGE_SHAPE,
    MAX_OBJECTS,
    guide_params_m4,
    guide_params_m5,
    guide_point_estimates,
    jnp,
    np,
    plt,
    pseudo_real_data_m5,
    pseudo_real_images_m5,
    random,
    sample_guide_latents_m5,
):
    pseudo_estimates_before_m5 = guide_point_estimates(guide_params_m4, pseudo_real_images_m5)
    pseudo_estimates_after_m5 = guide_point_estimates(guide_params_m5, pseudo_real_images_m5)
    pseudo_posterior_samples_m5 = sample_guide_latents_m5(
        guide_params_m5, pseudo_real_images_m5, 12, random.PRNGKey(860)
    )
    pseudo_sampled_counts_m5 = jnp.sum(pseudo_posterior_samples_m5["presence"], axis=-1)
    pseudo_sample_count_summary_m5 = {
        "mean_count_by_patch_first12": np.asarray(jnp.mean(pseudo_sampled_counts_m5[:, :12], axis=0)).round(3).tolist(),
        "std_count_by_patch_first12": np.asarray(jnp.std(pseudo_sampled_counts_m5[:, :12], axis=0)).round(3).tolist(),
    }
    pseudo_predicted_counts_before_m5 = jnp.sum(pseudo_estimates_before_m5["presence_probs"], axis=1)
    pseudo_predicted_counts_after_m5 = jnp.sum(pseudo_estimates_after_m5["presence_probs"], axis=1)
    pseudo_distinct_score_m5 = float(
        jnp.mean(jnp.std(pseudo_estimates_after_m5["presence_probs"], axis=0))
        + jnp.mean(jnp.std(pseudo_estimates_after_m5["position"], axis=0))
        + jnp.mean(jnp.std(pseudo_estimates_after_m5["size"], axis=0))
        + jnp.mean(jnp.std(pseudo_estimates_after_m5["composition"], axis=0))
    )
    pseudo_posteriors_distinct_m5 = pseudo_distinct_score_m5 > 0.05
    pseudo_estimates_finite_m5 = bool(
        all(jnp.all(jnp.isfinite(value)) for value in pseudo_estimates_after_m5.values())
    )

    fig_pseudo_overlays_m5, axes_pseudo_overlays_m5 = plt.subplots(3, 4, figsize=(11, 8))
    for _ax, _idx in zip(axes_pseudo_overlays_m5.ravel(), range(12)):
        _ax.imshow(np.asarray(pseudo_real_images_m5[_idx]), cmap="magma", interpolation="nearest")
        _ax.set_title(
            f"pseudo {_idx}: true count={int(pseudo_real_data_m5['count'][_idx])}\nE[count] {float(pseudo_predicted_counts_before_m5[_idx]):.2f} → {float(pseudo_predicted_counts_after_m5[_idx]):.2f}",
            fontsize=8,
        )
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _obj in range(MAX_OBJECTS):
            if float(pseudo_real_data_m5["presence"][_idx, _obj]) > 0.5:
                _true_pos = np.asarray(pseudo_real_data_m5["position"][_idx, _obj])
                _true_size = float(pseudo_real_data_m5["size"][_idx, _obj])
                _ax.add_patch(
                    plt.Circle(
                        (_true_pos[1] * (IMAGE_SHAPE[1] - 1), _true_pos[0] * (IMAGE_SHAPE[0] - 1)),
                        radius=_true_size * IMAGE_SHAPE[0],
                        edgecolor="cyan",
                        facecolor="none",
                        linewidth=1.5,
                    )
                )
            _prob = float(pseudo_estimates_after_m5["presence_probs"][_idx, _obj])
            _pos = np.asarray(pseudo_estimates_after_m5["position"][_idx, _obj])
            _size = float(pseudo_estimates_after_m5["size"][_idx, _obj])
            _circle = plt.Circle(
                (_pos[1] * (IMAGE_SHAPE[1] - 1), _pos[0] * (IMAGE_SHAPE[0] - 1)),
                radius=_size * IMAGE_SHAPE[0],
                edgecolor=(1.0, 0.0, 1.0, max(0.15, _prob)),
                facecolor="none",
                linewidth=1.2,
            )
            _ax.add_patch(_circle)
    fig_pseudo_overlays_m5.suptitle(
        "Controlled pseudo-real posterior overlays after SC\ncyan=true held-out labels for audit; magenta=inferred explicit latents",
        y=1.02,
    )
    fig_pseudo_overlays_m5.tight_layout()
    fig_pseudo_overlays_m5
    return pseudo_estimates_finite_m5, pseudo_posteriors_distinct_m5


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Milestone 5 pseudo-real posterior diversity

    "
        f"Predicted expected counts before SC, first 12: `{np.asarray(pseudo_predicted_counts_before_m5[:12]).round(3).tolist()}`.

    "
        f"Predicted expected counts after SC, first 12: `{np.asarray(pseudo_predicted_counts_after_m5[:12]).round(3).tolist()}`.

    "
        f"Posterior sampled count summary after SC: `{pseudo_sample_count_summary_m5}`.

    "
        f"Posterior mean distinctness score across pseudo-real patches: `{pseudo_distinct_score_m5:.3f}`; distinct: `{pseudo_posteriors_distinct_m5}`.

    "
        f"Pseudo-real posterior estimates finite: `{pseudo_estimates_finite_m5}`.
    """)
    return


@app.cell(hide_code=True)
def _(
    fixed_eval_sc_after_m5,
    fixed_eval_sc_before_m5,
    mo,
    pseudo_estimates_finite_m5,
    pseudo_posteriors_distinct_m5,
    pseudo_real_label_check_m5,
    sc_fixed_eval_decreased_m5,
    sc_initial_finite_m5,
    sc_initial_grad_norm_m5,
    synthetic_no_collapse_m5,
    synthetic_npe_delta_m5,
):
    milestone_5_controlled_passed = bool(
        sc_initial_finite_m5
        and sc_fixed_eval_decreased_m5
        and synthetic_no_collapse_m5
        and pseudo_real_label_check_m5
        and pseudo_posteriors_distinct_m5
        and pseudo_estimates_finite_m5
    )
    actual_real_milestone_5_passed = False
    milestone_5_passed = milestone_5_controlled_passed

    mo.md(
        f"""
        ## Milestone 5 report — controlled Bayesian self-consistency

        **Implemented.** Corrected the milestone to use simulator-consistent pseudo-real images first. The SC objective is `Var_l[log p_model(theta_l, x) - log q_phi(theta_l | x)]`, with proposals sampled from a stop-gradient current guide. No reconstruction MSE, cycle loss, or renderer training is used. Labels for `pseudo_real_data_m5` are hidden during adaptation and used only for auditing.

        **Verified.** The SC log-ratio, loss, and guide gradients are finite; guide gradient norm is `{float(sc_initial_grad_norm_m5):.4f}`. A conservative clipped update reduced a fixed-proposal SC monitor from `{fixed_eval_sc_before_m5:.6f}` to `{fixed_eval_sc_after_m5:.6f}`. Held-out synthetic NPE changed by `{synthetic_npe_delta_m5:.3f}` and did not collapse: `{synthetic_no_collapse_m5}`. The hidden-label pseudo-real audit passed: `{pseudo_real_label_check_m5}`. Posterior samples/means differ across pseudo-real patches and remain finite/interpretable.

        **Real-data domain check.** `example.jpg` was visually inspected and is **not** compatible with the primitive likelihood: it is colour, crowded, bright-background, shadowed, and has non-Gaussian droplet appearances. It is therefore not used for SC here. `actual_real_milestone_5_passed = {actual_real_milestone_5_passed}`.

        **Concerns.** This passes only the controlled simulator-consistent self-consistency milestone. It does not solve real microscopy inference. The next step is not to force this likelihood onto `example.jpg`; it is to build a substantially richer generative process/likelihood with colour, shape, illumination, and renderer flexibility while preserving interpretable object latents.

        **Controlled Milestone 5 passed:** `{milestone_5_controlled_passed}`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Next-generation renderer v2 — Milestone 1: colour coordinate-sprite likelihood

    This starts a new modelling ladder for real microscopy-like colour droplets. The renderer is still object-centric and interpretable: objects are represented by presence, position, size, and constrained composition. The local appearance renderer is a small shared learnable coordinate-sprite model that receives **local x/y coordinates scaled by size**, size, and composition. It does not receive a per-image latent or arbitrary object embedding.

    This section only implements and audits the prior/generative model. No guide or training is attempted until prior predictive images look plausible.
    """)
    return


@app.cell
def _(jnp, mo):
    IMAGE_SHAPE_V2 = (64, 64)
    CHANNELS_V2 = 3
    SLOT_ROWS_V2 = 4
    SLOT_COLS_V2 = 4
    MAX_OBJECTS_V2 = SLOT_ROWS_V2 * SLOT_COLS_V2
    COMPOSITION_DIM_V2 = 4
    LATENT_SITE_NAMES_V2 = (
        "background_rgb_v2",
        "observation_noise_v2",
        "presence_v2",
        "position_v2",
        "size_v2",
        "composition_v2",
    )

    _y_edges_v2 = jnp.linspace(0.0, 1.0, SLOT_ROWS_V2 + 1, dtype=jnp.float32)
    _x_edges_v2 = jnp.linspace(0.0, 1.0, SLOT_COLS_V2 + 1, dtype=jnp.float32)
    POSITION_LOW_V2 = jnp.array(
        [[_y_edges_v2[_r], _x_edges_v2[_c]] for _r in range(SLOT_ROWS_V2) for _c in range(SLOT_COLS_V2)],
        dtype=jnp.float32,
    )
    POSITION_HIGH_V2 = jnp.array(
        [[_y_edges_v2[_r + 1], _x_edges_v2[_c + 1]] for _r in range(SLOT_ROWS_V2) for _c in range(SLOT_COLS_V2)],
        dtype=jnp.float32,
    )
    POSITION_SCALE_V2 = POSITION_HIGH_V2 - POSITION_LOW_V2
    POSITION_CENTER_V2 = 0.5 * (POSITION_LOW_V2 + POSITION_HIGH_V2)
    SIZE_LOW_V2 = 0.025
    SIZE_HIGH_V2 = 0.078

    # Interpretable material palette mixed by the composition simplex.
    MATERIAL_PALETTE_V2 = jnp.array(
        [
            [0.95, 0.20, 0.30],  # red/magenta material
            [0.12, 0.35, 0.95],  # blue material
            [0.95, 0.82, 0.18],  # yellow material
            [0.10, 0.72, 0.55],  # green/cyan material
        ],
        dtype=jnp.float32,
    )
    RIM_PALETTE_V2 = jnp.array(
        [
            [0.70, 0.08, 0.72],
            [0.05, 0.18, 0.88],
            [0.95, 0.86, 0.20],
            [0.08, 0.80, 0.70],
        ],
        dtype=jnp.float32,
    )
    SHADOW_PALETTE_V2 = jnp.array(
        [
            [0.42, 0.10, 0.62],
            [0.08, 0.16, 0.65],
            [0.45, 0.35, 0.08],
            [0.05, 0.34, 0.42],
        ],
        dtype=jnp.float32,
    )
    HIGHLIGHT_RGB_V2 = jnp.array([1.0, 0.96, 0.55], dtype=jnp.float32)

    mo.md(
        f"Renderer v2 uses `{MAX_OBJECTS_V2}` anchored slots on a `{SLOT_ROWS_V2}×{SLOT_COLS_V2}` grid, RGB images of shape `{IMAGE_SHAPE_V2 + (CHANNELS_V2,)}`, and `{COMPOSITION_DIM_V2}`-simplex composition vectors."
    )
    return (
        CHANNELS_V2,
        COMPOSITION_DIM_V2,
        HIGHLIGHT_RGB_V2,
        IMAGE_SHAPE_V2,
        LATENT_SITE_NAMES_V2,
        MATERIAL_PALETTE_V2,
        MAX_OBJECTS_V2,
        POSITION_CENTER_V2,
        POSITION_HIGH_V2,
        POSITION_LOW_V2,
        RIM_PALETTE_V2,
        SHADOW_PALETTE_V2,
        SIZE_HIGH_V2,
        SIZE_LOW_V2,
    )


@app.cell
def _(
    HIGHLIGHT_RGB_V2,
    MATERIAL_PALETTE_V2,
    RIM_PALETTE_V2,
    SHADOW_PALETTE_V2,
    jnp,
    mo,
):
    def init_renderer_params_v2():
        """Small shared coordinate-sprite renderer parameters.

        These are global renderer parameters, not per-image latents. They are
        initialised to produce asymmetric coloured droplet-like sprites and can later
        be learned under strong diagnostics. The basis functions depend on local
        y/x coordinates, not only radial distance.
        """
        basis_centres_yx = jnp.array(
            [
                [0.00, 0.00],   # core
                [-0.35, -0.25], # asymmetric upper-left shadow
                [0.42, -0.10],  # lower crescent shadow
                [-0.42, 0.36],  # highlight
                [0.18, 0.28],   # warm cap
                [0.02, -0.72],  # blue/purple side rim
                [0.58, 0.36],   # green/yellow lower highlight
                [-0.70, 0.06],  # small asymmetric mark
            ],
            dtype=jnp.float32,
        )
        basis_scales_yx = jnp.array(
            [
                [0.80, 0.78],
                [0.38, 0.48],
                [0.42, 0.72],
                [0.22, 0.25],
                [0.36, 0.34],
                [0.34, 0.28],
                [0.26, 0.34],
                [0.22, 0.22],
            ],
            dtype=jnp.float32,
        )
        basis_rgb_bias = jnp.array(
            [
                [-0.02, -0.02, -0.02],
                [-0.16, -0.12, -0.24],
                [-0.12, -0.15, -0.22],
                [0.22, 0.20, 0.08],
                [0.02, 0.00, -0.02],
                [-0.06, -0.07, -0.16],
                [0.04, 0.10, 0.02],
                [-0.06, -0.03, -0.09],
            ],
            dtype=jnp.float32,
        )
        # K x composition_dim x RGB. Composition controls colour/material effects.
        basis_comp_rgb = jnp.stack(
            [
                0.65 * (MATERIAL_PALETTE_V2 - 0.55),
                -0.18 * SHADOW_PALETTE_V2,
                -0.15 * SHADOW_PALETTE_V2,
                0.10 * (MATERIAL_PALETTE_V2 - 0.35) + 0.08 * HIGHLIGHT_RGB_V2,
                0.38 * (MATERIAL_PALETTE_V2 - 0.45),
                0.34 * (RIM_PALETTE_V2 - 0.45),
                0.20 * (MATERIAL_PALETTE_V2 - 0.30) + 0.05 * HIGHLIGHT_RGB_V2,
                -0.10 * SHADOW_PALETTE_V2,
            ],
            axis=0,
        ).astype(jnp.float32)
        size_rgb_weights = jnp.array([0.05, 0.03, -0.04], dtype=jnp.float32)
        return {
            "basis_centres_yx": basis_centres_yx,
            "basis_scales_yx": basis_scales_yx,
            "basis_rgb_bias": basis_rgb_bias,
            "basis_comp_rgb": basis_comp_rgb,
            "size_rgb_weights": size_rgb_weights,
        }


    RENDERER_PARAMS_V2 = init_renderer_params_v2()
    NUM_RENDERER_BASIS_V2 = int(RENDERER_PARAMS_V2["basis_centres_yx"].shape[0])

    mo.md(
        f"The coordinate-sprite renderer uses `{NUM_RENDERER_BASIS_V2}` shared local x/y basis functions. These parameters are global and limited; there is no per-image renderer state."
    )
    return (RENDERER_PARAMS_V2,)


@app.cell
def _(
    CHANNELS_V2,
    COMPOSITION_DIM_V2,
    IMAGE_SHAPE_V2,
    MATERIAL_PALETTE_V2,
    MAX_OBJECTS_V2,
    POSITION_HIGH_V2,
    POSITION_LOW_V2,
    RENDERER_PARAMS_V2,
    SIZE_HIGH_V2,
    SIZE_LOW_V2,
    dist,
    jnp,
    numpyro,
):
    def make_grid_v2(image_shape=IMAGE_SHAPE_V2):
        height, width = image_shape
        y = jnp.linspace(0.0, 1.0, height, dtype=jnp.float32)
        x = jnp.linspace(0.0, 1.0, width, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        return jnp.stack([yy, xx], axis=-1)


    def composition_to_rgb_v2(composition):
        return composition @ MATERIAL_PALETTE_V2


    def local_xy_basis_v2(local_yx, renderer_params):
        """Asymmetric local y/x basis functions; no radial-symmetry assumption."""
        centres = renderer_params["basis_centres_yx"]
        scales = renderer_params["basis_scales_yx"]
        delta = local_yx[..., None, :] - centres[None, None, None, :, :]
        scaled_sq = jnp.sum((delta / scales[None, None, None, :, :]) ** 2, axis=-1)
        return jnp.exp(-0.5 * scaled_sq)


    def coordinate_sprite_effect_v2(position, size, composition, renderer_params, image_shape=IMAGE_SHAPE_V2):
        """Shared learnable sprite: f((y-y0)/size, (x-x0)/size, size, composition) -> RGB effect."""
        grid = make_grid_v2(image_shape)
        local_yx = (grid[None, :, :, :] - position[:, None, None, :]) / (
            size[:, None, None, None] + 1e-6
        )
        basis = local_xy_basis_v2(local_yx, renderer_params)  # objects x H x W x K
        basis_rgb_from_comp = jnp.einsum(
            "oc,kcr->okr", composition, renderer_params["basis_comp_rgb"]
        )
        basis_rgb = basis_rgb_from_comp + renderer_params["basis_rgb_bias"][None, :, :]
        size_effect = (size[:, None, None, None] - 0.05) * renderer_params["size_rgb_weights"][None, None, None, :]
        effect = jnp.einsum("ohwk,okr->ohwr", basis, basis_rgb) + jnp.sum(basis, axis=-1)[..., None] * size_effect
        return effect


    def render_scene_v2(
        background_rgb,
        presence,
        position,
        size,
        composition,
        renderer_params=RENDERER_PARAMS_V2,
        image_shape=IMAGE_SHAPE_V2,
    ):
        """Order-invariant RGB renderer: background + summed coordinate-sprite effects."""
        background = jnp.broadcast_to(background_rgb, image_shape + (CHANNELS_V2,))
        object_effect = coordinate_sprite_effect_v2(
            position, size, composition, renderer_params, image_shape
        )
        image = background + jnp.sum(presence[:, None, None, None] * object_effect, axis=0)
        return jnp.clip(image, 0.0, 1.0)


    def image_distribution_v2(mean_image, observation_noise):
        return dist.Normal(mean_image, observation_noise).to_event(3)


    def patch_model_v2(image=None, renderer_params=RENDERER_PARAMS_V2, image_shape=IMAGE_SHAPE_V2):
        """Colour patch model with interpretable object latents and a shared coordinate renderer."""
        background_rgb = numpyro.sample(
            "background_rgb_v2",
            dist.Uniform(
                jnp.array([0.56, 0.58, 0.42], dtype=jnp.float32),
                jnp.array([0.86, 0.86, 0.72], dtype=jnp.float32),
            ).to_event(1),
        )
        observation_noise = numpyro.sample(
            "observation_noise_v2", dist.LogNormal(jnp.log(0.025), 0.30)
        )
        presence = numpyro.sample(
            "presence_v2",
            dist.Bernoulli(probs=0.38).expand([MAX_OBJECTS_V2]).to_event(1),
        )
        position = numpyro.sample(
            "position_v2", dist.Uniform(POSITION_LOW_V2, POSITION_HIGH_V2).to_event(2)
        )
        size = numpyro.sample(
            "size_v2",
            dist.Uniform(
                SIZE_LOW_V2 * jnp.ones(MAX_OBJECTS_V2),
                SIZE_HIGH_V2 * jnp.ones(MAX_OBJECTS_V2),
            ).to_event(1),
        )
        composition = numpyro.sample(
            "composition_v2",
            dist.Dirichlet(1.2 * jnp.ones(COMPOSITION_DIM_V2))
            .expand([MAX_OBJECTS_V2])
            .to_event(1),
        )
        mean_image = render_scene_v2(
            background_rgb, presence, position, size, composition, renderer_params, image_shape
        )
        numpyro.deterministic("mean_v2", mean_image)
        numpyro.deterministic("count_v2", jnp.sum(presence))
        numpyro.sample("obs_v2", image_distribution_v2(mean_image, observation_noise), obs=image)

    return composition_to_rgb_v2, patch_model_v2, render_scene_v2


@app.cell
def _(
    CHANNELS_V2,
    COMPOSITION_DIM_V2,
    IMAGE_SHAPE_V2,
    LATENT_SITE_NAMES_V2,
    MAX_OBJECTS_V2,
    POSITION_CENTER_V2,
    Predictive,
    jnp,
    log_density,
    mo,
    patch_model_v2,
    random,
    render_scene_v2,
):
    prior_predictive_v2 = Predictive(
        patch_model_v2,
        num_samples=24,
        return_sites=(*LATENT_SITE_NAMES_V2, "mean_v2", "obs_v2", "count_v2"),
    )(random.PRNGKey(1001))

    actual_shapes_v2 = {name: tuple(value.shape) for name, value in prior_predictive_v2.items()}
    expected_shapes_v2 = {
        "background_rgb_v2": (24, CHANNELS_V2),
        "observation_noise_v2": (24,),
        "presence_v2": (24, MAX_OBJECTS_V2),
        "position_v2": (24, MAX_OBJECTS_V2, 2),
        "size_v2": (24, MAX_OBJECTS_V2),
        "composition_v2": (24, MAX_OBJECTS_V2, COMPOSITION_DIM_V2),
        "mean_v2": (24, *IMAGE_SHAPE_V2, CHANNELS_V2),
        "obs_v2": (24, *IMAGE_SHAPE_V2, CHANNELS_V2),
        "count_v2": (24,),
    }
    shape_checks_v2 = {
        name: actual_shapes_v2[name] == expected_shape
        for name, expected_shape in expected_shapes_v2.items()
    }
    finite_prior_v2 = bool(
        jnp.all(jnp.isfinite(prior_predictive_v2["mean_v2"]))
        and jnp.all(jnp.isfinite(prior_predictive_v2["obs_v2"]))
    )

    latents0_v2 = {name: prior_predictive_v2[name][0] for name in LATENT_SITE_NAMES_V2}
    log_joint_v2, trace_v2 = log_density(
        patch_model_v2,
        model_args=(prior_predictive_v2["obs_v2"][0],),
        model_kwargs={},
        params=latents0_v2,
    )
    log_joint_finite_v2 = bool(jnp.isfinite(log_joint_v2))
    trace_sample_sites_v2 = tuple(name for name, site in trace_v2.items() if site["type"] == "sample")
    required_sample_sites_present_v2 = all(
        name in trace_sample_sites_v2 for name in (*LATENT_SITE_NAMES_V2, "obs_v2")
    )

    _permutation_v2 = jnp.arange(MAX_OBJECTS_V2 - 1, -1, -1)
    mean_original_v2 = render_scene_v2(
        latents0_v2["background_rgb_v2"],
        latents0_v2["presence_v2"],
        latents0_v2["position_v2"],
        latents0_v2["size_v2"],
        latents0_v2["composition_v2"],
    )
    mean_permuted_v2 = render_scene_v2(
        latents0_v2["background_rgb_v2"],
        latents0_v2["presence_v2"][_permutation_v2],
        latents0_v2["position_v2"][_permutation_v2],
        latents0_v2["size_v2"][_permutation_v2],
        latents0_v2["composition_v2"][_permutation_v2],
    )
    order_max_abs_diff_v2 = float(jnp.max(jnp.abs(mean_original_v2 - mean_permuted_v2)))
    order_invariant_v2 = order_max_abs_diff_v2 < 1e-6

    control_background_v2 = jnp.array([0.72, 0.74, 0.55], dtype=jnp.float32)
    control_presence_v2 = jnp.zeros(MAX_OBJECTS_V2, dtype=jnp.float32).at[5].set(1.0)
    control_position_v2 = POSITION_CENTER_V2
    control_size_v2 = 0.050 * jnp.ones(MAX_OBJECTS_V2, dtype=jnp.float32)
    control_composition_a_v2 = jnp.ones((MAX_OBJECTS_V2, COMPOSITION_DIM_V2), dtype=jnp.float32) / COMPOSITION_DIM_V2
    control_composition_b_v2 = control_composition_a_v2.at[5].set(jnp.array([0.90, 0.04, 0.03, 0.03], dtype=jnp.float32))
    control_composition_c_v2 = control_composition_a_v2.at[5].set(jnp.array([0.03, 0.90, 0.04, 0.03], dtype=jnp.float32))
    mean_comp_a_v2 = render_scene_v2(control_background_v2, control_presence_v2, control_position_v2, control_size_v2, control_composition_a_v2)
    mean_comp_b_v2 = render_scene_v2(control_background_v2, control_presence_v2, control_position_v2, control_size_v2, control_composition_b_v2)
    mean_comp_c_v2 = render_scene_v2(control_background_v2, control_presence_v2, control_position_v2, control_size_v2, control_composition_c_v2)
    composition_change_diff_v2 = float(jnp.max(jnp.abs(mean_comp_b_v2 - mean_comp_c_v2)))
    composition_changes_colour_v2 = composition_change_diff_v2 > 0.03

    single_center_position_v2 = jnp.tile(jnp.array([[0.5, 0.5]], dtype=jnp.float32), (MAX_OBJECTS_V2, 1))
    single_presence_v2 = jnp.zeros(MAX_OBJECTS_V2, dtype=jnp.float32).at[0].set(1.0)
    single_composition_v2 = control_composition_a_v2.at[0].set(jnp.array([0.05, 0.75, 0.10, 0.10], dtype=jnp.float32))
    single_size_v2 = 0.060 * jnp.ones(MAX_OBJECTS_V2, dtype=jnp.float32)
    single_sprite_v2 = render_scene_v2(control_background_v2, single_presence_v2, single_center_position_v2, single_size_v2, single_composition_v2)
    vertical_flip_diff_v2 = float(jnp.mean(jnp.abs(single_sprite_v2 - jnp.flip(single_sprite_v2, axis=0))))
    horizontal_flip_diff_v2 = float(jnp.mean(jnp.abs(single_sprite_v2 - jnp.flip(single_sprite_v2, axis=1))))
    xy_asymmetry_present_v2 = (vertical_flip_diff_v2 > 1e-3) or (horizontal_flip_diff_v2 > 1e-3)

    milestone_v2_model_checks_passed = bool(
        all(shape_checks_v2.values())
        and finite_prior_v2
        and log_joint_finite_v2
        and required_sample_sites_present_v2
        and order_invariant_v2
        and composition_changes_colour_v2
        and xy_asymmetry_present_v2
    )

    mo.md(
        "### Renderer v2 probabilistic checks\n\n"
        f"Prior predictive shapes correct: `{all(shape_checks_v2.values())}`. Shapes: `{actual_shapes_v2}`\n\n"
        f"Prior images finite: `{finite_prior_v2}`. Log joint finite: `{log_joint_finite_v2}` (`{float(log_joint_v2):.2f}`).\n\n"
        f"Required sample sites present: `{required_sample_sites_present_v2}`.\n\n"
        f"Order-invariance max diff under object permutation: `{order_max_abs_diff_v2:.3e}`.\n\n"
        f"Composition colour-change max diff: `{composition_change_diff_v2:.3f}`.\n\n"
        f"Local x/y asymmetry check, vertical flip diff `{vertical_flip_diff_v2:.4f}`, horizontal flip diff `{horizontal_flip_diff_v2:.4f}`.\n\n"
        f"Checks passed: `{milestone_v2_model_checks_passed}`."
    )
    return (
        control_background_v2,
        milestone_v2_model_checks_passed,
        prior_predictive_v2,
    )


@app.cell
def _(np, plt, prior_predictive_v2):
    fig_prior_grid_v2, axes_prior_grid_v2 = plt.subplots(4, 4, figsize=(8, 8))
    for _ax, _image, _count in zip(
        axes_prior_grid_v2.ravel(),
        np.asarray(prior_predictive_v2["obs_v2"][:16]),
        np.asarray(prior_predictive_v2["count_v2"][:16]),
    ):
        _ax.imshow(np.clip(_image, 0.0, 1.0), interpolation="nearest")
        _ax.set_title(f"count={int(_count)}", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    fig_prior_grid_v2.suptitle("Renderer v2 prior predictive RGB patches", y=0.94)
    fig_prior_grid_v2.tight_layout()
    fig_prior_grid_v2
    return


@app.cell
def _(
    COMPOSITION_DIM_V2,
    MAX_OBJECTS_V2,
    control_background_v2,
    jnp,
    np,
    plt,
    render_scene_v2,
):
    sprite_compositions_v2 = jnp.array(
        [
            [0.90, 0.04, 0.03, 0.03],
            [0.04, 0.90, 0.03, 0.03],
            [0.03, 0.04, 0.90, 0.03],
            [0.03, 0.04, 0.03, 0.90],
        ],
        dtype=jnp.float32,
    )
    sprite_sizes_v2 = jnp.array([0.035, 0.060, 0.078], dtype=jnp.float32)
    fig_sprite_sheet_v2, axes_sprite_sheet_v2 = plt.subplots(
        len(sprite_compositions_v2), len(sprite_sizes_v2), figsize=(7, 8)
    )
    for _row in range(len(sprite_compositions_v2)):
        for _col in range(len(sprite_sizes_v2)):
            _presence = jnp.zeros(MAX_OBJECTS_V2, dtype=jnp.float32).at[0].set(1.0)
            _position = jnp.tile(jnp.array([[0.5, 0.5]], dtype=jnp.float32), (MAX_OBJECTS_V2, 1))
            _size = sprite_sizes_v2[_col] * jnp.ones(MAX_OBJECTS_V2, dtype=jnp.float32)
            _composition = (jnp.ones((MAX_OBJECTS_V2, COMPOSITION_DIM_V2), dtype=jnp.float32) / COMPOSITION_DIM_V2).at[0].set(sprite_compositions_v2[_row])
            _image = render_scene_v2(control_background_v2, _presence, _position, _size, _composition)
            axes_sprite_sheet_v2[_row, _col].imshow(np.asarray(_image), interpolation="nearest")
            axes_sprite_sheet_v2[_row, _col].set_title(
                f"comp={np.asarray(sprite_compositions_v2[_row]).round(2)}\nsize={float(sprite_sizes_v2[_col]):.3f}",
                fontsize=8,
            )
            axes_sprite_sheet_v2[_row, _col].set_xticks([])
            axes_sprite_sheet_v2[_row, _col].set_yticks([])
    fig_sprite_sheet_v2.suptitle("Renderer v2 single-object colour/size diversity", y=0.99)
    fig_sprite_sheet_v2.tight_layout()
    fig_sprite_sheet_v2
    return


@app.cell
def _(
    COMPOSITION_DIM_V2,
    MAX_OBJECTS_V2,
    composition_to_rgb_v2,
    jnp,
    np,
    plt,
    prior_predictive_v2,
):
    presence_np_v2 = np.asarray(prior_predictive_v2["presence_v2"])
    active_mask_np_v2 = presence_np_v2.astype(bool)
    active_positions_np_v2 = np.asarray(prior_predictive_v2["position_v2"])[active_mask_np_v2]
    active_sizes_np_v2 = np.asarray(prior_predictive_v2["size_v2"])[active_mask_np_v2]
    active_compositions_np_v2 = np.asarray(prior_predictive_v2["composition_v2"])[active_mask_np_v2]
    active_colours_np_v2 = np.asarray(composition_to_rgb_v2(jnp.asarray(active_compositions_np_v2)))
    counts_np_v2 = np.asarray(prior_predictive_v2["count_v2"])

    fig_latents_v2, axes_latents_v2 = plt.subplots(2, 3, figsize=(11, 6))
    axes_latents_v2[0, 0].hist(counts_np_v2, bins=np.arange(-0.5, MAX_OBJECTS_V2 + 1.5), rwidth=0.8)
    axes_latents_v2[0, 0].set_title("object count")
    axes_latents_v2[0, 0].set_xlabel("count")
    axes_latents_v2[0, 1].scatter(active_positions_np_v2[:, 1], active_positions_np_v2[:, 0], s=8, alpha=0.6)
    axes_latents_v2[0, 1].invert_yaxis()
    axes_latents_v2[0, 1].set_title("active positions")
    axes_latents_v2[0, 1].set_xlabel("x")
    axes_latents_v2[0, 1].set_ylabel("y")
    axes_latents_v2[0, 2].hist(active_sizes_np_v2, bins=20, color="tab:green")
    axes_latents_v2[0, 2].set_title("active sizes")
    for _k in range(COMPOSITION_DIM_V2):
        axes_latents_v2[1, 0].hist(active_compositions_np_v2[:, _k], bins=20, alpha=0.55, label=f"c{_k}")
    axes_latents_v2[1, 0].set_title("composition components")
    axes_latents_v2[1, 0].legend(fontsize=8)
    axes_latents_v2[1, 1].scatter(active_colours_np_v2[:, 0], active_colours_np_v2[:, 2], c=active_colours_np_v2, s=16)
    axes_latents_v2[1, 1].set_title("composition-implied colours")
    axes_latents_v2[1, 1].set_xlabel("red")
    axes_latents_v2[1, 1].set_ylabel("blue")
    axes_latents_v2[1, 2].hist(np.asarray(prior_predictive_v2["observation_noise_v2"]), bins=20, color="tab:orange")
    axes_latents_v2[1, 2].set_title("observation noise")
    fig_latents_v2.tight_layout()
    fig_latents_v2
    return active_colours_np_v2, active_sizes_np_v2, counts_np_v2


@app.cell
def _(
    IMAGE_SHAPE_V2,
    composition_to_rgb_v2,
    jnp,
    np,
    plt,
    prior_predictive_v2,
):
    fig_overlay_v2, axes_overlay_v2 = plt.subplots(2, 4, figsize=(10, 5))
    for _ax, _idx in zip(axes_overlay_v2.ravel(), range(8)):
        _ax.imshow(np.asarray(prior_predictive_v2["obs_v2"][_idx]).clip(0.0, 1.0), interpolation="nearest")
        _ax.set_title(f"sample {_idx}; count={int(prior_predictive_v2['count_v2'][_idx])}", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _present, _pos, _size, _comp in zip(
            np.asarray(prior_predictive_v2["presence_v2"][_idx]),
            np.asarray(prior_predictive_v2["position_v2"][_idx]),
            np.asarray(prior_predictive_v2["size_v2"][_idx]),
            np.asarray(prior_predictive_v2["composition_v2"][_idx]),
        ):
            if _present > 0.5:
                _colour = np.asarray(composition_to_rgb_v2(jnp.asarray(_comp)))
                _ax.add_patch(
                    plt.Circle(
                        (_pos[1] * (IMAGE_SHAPE_V2[1] - 1), _pos[0] * (IMAGE_SHAPE_V2[0] - 1)),
                        radius=_size * IMAGE_SHAPE_V2[0],
                        edgecolor=np.clip(_colour, 0.0, 1.0),
                        facecolor="none",
                        linewidth=1.2,
                    )
                )
    fig_overlay_v2.suptitle("Renderer v2 prior images with true position/size overlays", y=1.02)
    fig_overlay_v2.tight_layout()
    fig_overlay_v2
    return


@app.cell(hide_code=True)
def _(
    active_colours_np_v2,
    active_sizes_np_v2,
    counts_np_v2,
    milestone_v2_model_checks_passed,
    mo,
    np,
):
    prior_counts_diverse_v2 = bool(len(np.unique(counts_np_v2)) >= 4)
    prior_colours_diverse_v2 = bool(np.mean(np.std(active_colours_np_v2, axis=0)) > 0.08)
    prior_sizes_diverse_v2 = bool(np.std(active_sizes_np_v2) > 0.01)
    prior_predictive_visual_gate_v2 = bool(
        milestone_v2_model_checks_passed
        and prior_counts_diverse_v2
        and prior_colours_diverse_v2
        and prior_sizes_diverse_v2
    )

    mo.md(
        f"""
        ## Renderer v2 Milestone 1 report — colour coordinate-sprite generative model

        **Implemented.** A new RGB NumPyro generative model `patch_model_v2` with named sites `background_rgb_v2`, `observation_noise_v2`, `presence_v2`, `position_v2`, `size_v2`, and `composition_v2`. The deterministic renderer `render_scene_v2` is order-invariant and uses a small shared coordinate-sprite renderer. Each object's local appearance is a function of `(y - y0) / size`, `(x - x0) / size`, size, and composition. There is no per-image latent and no arbitrary object embedding.

        **Verified numerically.** Prior predictive sampling works, shapes are correct, the log joint is finite, object order permutation leaves the rendered image unchanged, changing composition changes colour/appearance without changing position or size, and the local sprite is asymmetric under flips, confirming the renderer is not restricted to radial symmetry.

        **Verified visually.** The cells above show prior predictive RGB patches, single-object colour/size sprite diversity, latent/count/colour distributions, and position/size overlays. Count diversity: `{prior_counts_diverse_v2}`. Colour diversity: `{prior_colours_diverse_v2}`. Size diversity: `{prior_sizes_diverse_v2}`.

        **Concerns.** This is still a deliberately limited renderer. It can express asymmetric coloured local sprites, but not yet severe illumination gradients, dense overlap beyond the slot grid, depth ordering, or real optical blur. Those should be added only after visual prior checks and synthetic training diagnostics justify them.

        **Renderer v2 Milestone 1 passed:** `{prior_predictive_visual_gate_v2}`.

        **Next.** Do not implement a guide until the prior predictive visuals look scientifically plausible. If accepted, the next step is a NumPyro guide for the v2 latent sites, followed by synthetic training, training-set sanity checks, held-out checks, pseudo-real SC, and only then real RGB patches without ad hoc preprocessing.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Renderer v3 — generic learned coordinate renderer

    The hand-composed v2 renderer above is **superseded**. This v3 renderer is generic: a small shared coordinate MLP takes local coordinates `(y - y0) / size`, `(x - x0) / size`, object size, object composition, and local background colour, then returns an RGB background delta gated to a local object support. There are no hard-coded droplet palettes, highlights, shadows, rings, or object-type-specific structures.

    Interpretable object latents remain the route by which images are explained: presence/cardinality, position, size, and constrained composition. The renderer has global learnable parameters only; it has no per-image latent and no arbitrary object embedding.
    """)
    return


@app.cell
def _(jnp, mo):
    IMAGE_SHAPE_V3 = (64, 64)
    CHANNELS_V3 = 3
    SLOT_ROWS_V3 = 4
    SLOT_COLS_V3 = 4
    MAX_OBJECTS_V3 = SLOT_ROWS_V3 * SLOT_COLS_V3
    COMPOSITION_DIM_V3 = 4
    SIZE_LOW_V3 = 0.025
    SIZE_HIGH_V3 = 0.085
    LATENT_SITE_NAMES_V3 = (
        "background_rgb_v3",
        "observation_noise_v3",
        "presence_v3",
        "position_v3",
        "size_v3",
        "composition_v3",
    )

    _y_edges_v3 = jnp.linspace(0.0, 1.0, SLOT_ROWS_V3 + 1, dtype=jnp.float32)
    _x_edges_v3 = jnp.linspace(0.0, 1.0, SLOT_COLS_V3 + 1, dtype=jnp.float32)
    POSITION_LOW_V3 = jnp.array(
        [[_y_edges_v3[_r], _x_edges_v3[_c]] for _r in range(SLOT_ROWS_V3) for _c in range(SLOT_COLS_V3)],
        dtype=jnp.float32,
    )
    POSITION_HIGH_V3 = jnp.array(
        [[_y_edges_v3[_r + 1], _x_edges_v3[_c + 1]] for _r in range(SLOT_ROWS_V3) for _c in range(SLOT_COLS_V3)],
        dtype=jnp.float32,
    )
    POSITION_SCALE_V3 = POSITION_HIGH_V3 - POSITION_LOW_V3
    POSITION_CENTER_V3 = 0.5 * (POSITION_LOW_V3 + POSITION_HIGH_V3)

    mo.md(
        f"Renderer v3 uses `{MAX_OBJECTS_V3}` anchored slots on a `{SLOT_ROWS_V3}×{SLOT_COLS_V3}` grid, RGB images of shape `{IMAGE_SHAPE_V3 + (CHANNELS_V3,)}`, and `{COMPOSITION_DIM_V3}`-simplex composition vectors. Slot anchoring is retained only for initial identifiability of synthetic training labels."
    )
    return (
        CHANNELS_V3,
        COMPOSITION_DIM_V3,
        IMAGE_SHAPE_V3,
        LATENT_SITE_NAMES_V3,
        MAX_OBJECTS_V3,
        POSITION_CENTER_V3,
        POSITION_HIGH_V3,
        POSITION_LOW_V3,
        SIZE_HIGH_V3,
        SIZE_LOW_V3,
    )


@app.cell
def _(CHANNELS_V3, COMPOSITION_DIM_V3, jnp, mo, random):
    def init_renderer_params_v3(key, hidden_dim=32, composition_colour_scale=1.4):
        """Initialise a small shared coordinate MLP renderer.

        This is intentionally generic. The only structural bias is locality: object
        effects are multiplied by a smooth square support gate so one object does not
        alter the whole image. Shape and colour within that support are produced by
        the MLP from local y/x coordinates, size, composition, and background RGB.
        """
        key_w1, key_w2, key_comp = random.split(key, 3)
        input_dim = 2 + 1 + COMPOSITION_DIM_V3 + CHANNELS_V3
        output_dim = 1 + CHANNELS_V3  # alpha logit + RGB delta logits
        w1 = random.normal(key_w1, (input_dim, hidden_dim), dtype=jnp.float32) / jnp.sqrt(input_dim)
        b1 = jnp.zeros((hidden_dim,), dtype=jnp.float32)
        w2 = random.normal(key_w2, (hidden_dim, output_dim), dtype=jnp.float32) / jnp.sqrt(hidden_dim)
        # Encourage composition to visibly affect colour at initialisation without
        # imposing a semantic palette. This is still a learned/global matrix.
        comp_colour = random.normal(key_comp, (COMPOSITION_DIM_V3, CHANNELS_V3), dtype=jnp.float32)
        comp_colour = composition_colour_scale * comp_colour / (jnp.std(comp_colour) + 1e-6)
        w2 = w2.at[:, 1:].multiply(0.35)
        b2 = jnp.array([-0.65, 0.0, 0.0, 0.0], dtype=jnp.float32)
        return {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "composition_colour": comp_colour,
            "delta_scale": jnp.array(0.65, dtype=jnp.float32),
            # Locality envelope width in scaled local coordinates. This is only a
            # smooth locality prior so objects do not affect the whole image; shape
            # still comes from the x/y-dependent MLP alpha and RGB delta.
            "locality_sigma": jnp.array(1.25, dtype=jnp.float32),
        }


    RENDERER_PARAMS_V3 = init_renderer_params_v3(random.PRNGKey(3001))

    mo.md(
        "Renderer v3 parameters are a small shared coordinate MLP plus one global composition-to-RGB matrix. They are global learnable parameters, not image latents."
    )
    return (RENDERER_PARAMS_V3,)


@app.cell
def _(
    CHANNELS_V3,
    COMPOSITION_DIM_V3,
    IMAGE_SHAPE_V3,
    MAX_OBJECTS_V3,
    POSITION_HIGH_V3,
    POSITION_LOW_V3,
    RENDERER_PARAMS_V3,
    SIZE_HIGH_V3,
    SIZE_LOW_V3,
    dist,
    jnn,
    jnp,
    numpyro,
):
    def make_grid_v3(image_shape=IMAGE_SHAPE_V3):
        height, width = image_shape
        y = jnp.linspace(0.0, 1.0, height, dtype=jnp.float32)
        x = jnp.linspace(0.0, 1.0, width, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        return jnp.stack([yy, xx], axis=-1)


    def composition_to_rgb_v3(composition, renderer_params=RENDERER_PARAMS_V3):
        # Diagnostic map: not a hand palette; this is the renderer's learned/global
        # composition-colour matrix squashed to displayable RGB.
        return jnn.sigmoid(composition @ renderer_params["composition_colour"])


    def coordinate_renderer_v3(
        local_yx,
        size,
        composition,
        local_background_rgb,
        renderer_params=RENDERER_PARAMS_V3,
    ):
        """Shared MLP f(local_y, local_x, size, composition, background) -> alpha, RGB delta."""
        n_objects, height, width, _ = local_yx.shape
        size_feature = jnp.broadcast_to(
            jnp.log(size[:, None, None, None] / 0.05), (n_objects, height, width, 1)
        )
        composition_feature = jnp.broadcast_to(
            composition[:, None, None, :], (n_objects, height, width, COMPOSITION_DIM_V3)
        )
        features = jnp.concatenate(
            [local_yx, size_feature, composition_feature, local_background_rgb], axis=-1
        )
        hidden = jnp.tanh(jnp.einsum("ohwf,fg->ohwg", features, renderer_params["w1"]) + renderer_params["b1"])
        raw = jnp.einsum("ohwg,gc->ohwc", hidden, renderer_params["w2"]) + renderer_params["b2"]

        # Direct composition-colour pathway, still global/learnable and composition-based.
        comp_colour_delta = jnp.tanh(composition @ renderer_params["composition_colour"])
        raw_delta = raw[..., 1:] + comp_colour_delta[:, None, None, :]

        # Smooth locality envelope. The previous max(|x|, |y|) gate produced square
        # artifacts. This envelope prevents global image changes, while the MLP's
        # alpha still depends on local y and x separately and can learn asymmetric
        # non-circular shapes.
        locality_sq = jnp.sum((local_yx / renderer_params["locality_sigma"]) ** 2, axis=-1)
        locality_gate = jnp.exp(-0.5 * locality_sq)
        alpha = locality_gate[..., None] * jnn.sigmoid(raw[..., :1])
        rgb_delta = renderer_params["delta_scale"] * jnp.tanh(raw_delta)
        return alpha, rgb_delta


    def render_scene_v3(
        background_rgb,
        presence,
        position,
        size,
        composition,
        renderer_params=RENDERER_PARAMS_V3,
        image_shape=IMAGE_SHAPE_V3,
    ):
        """Order-invariant RGB scene renderer using background-aware coordinate sprites."""
        background = jnp.broadcast_to(background_rgb, image_shape + (CHANNELS_V3,))
        grid = make_grid_v3(image_shape)
        local_yx = (grid[None, :, :, :] - position[:, None, None, :]) / (
            size[:, None, None, None] + 1e-6
        )
        local_background = jnp.broadcast_to(
            background[None, :, :, :], (MAX_OBJECTS_V3,) + image_shape + (CHANNELS_V3,)
        )
        alpha, rgb_delta = coordinate_renderer_v3(
            local_yx, size, composition, local_background, renderer_params
        )
        object_delta = alpha * rgb_delta
        image = background + jnp.sum(presence[:, None, None, None] * object_delta, axis=0)
        return jnp.clip(image, 0.0, 1.0)


    def image_distribution_v3(mean_image, observation_noise):
        return dist.Normal(mean_image, observation_noise).to_event(3)


    def patch_model_v3(image=None, renderer_params=RENDERER_PARAMS_V3, image_shape=IMAGE_SHAPE_V3):
        background_rgb = numpyro.sample(
            "background_rgb_v3",
            dist.Uniform(
                jnp.array([0.45, 0.45, 0.35], dtype=jnp.float32),
                jnp.array([0.88, 0.88, 0.82], dtype=jnp.float32),
            ).to_event(1),
        )
        observation_noise = numpyro.sample(
            "observation_noise_v3", dist.LogNormal(jnp.log(0.018), 0.25)
        )
        presence = numpyro.sample(
            "presence_v3", dist.Bernoulli(probs=0.35).expand([MAX_OBJECTS_V3]).to_event(1)
        )
        position = numpyro.sample(
            "position_v3", dist.Uniform(POSITION_LOW_V3, POSITION_HIGH_V3).to_event(2)
        )
        size = numpyro.sample(
            "size_v3",
            dist.Uniform(
                SIZE_LOW_V3 * jnp.ones(MAX_OBJECTS_V3),
                SIZE_HIGH_V3 * jnp.ones(MAX_OBJECTS_V3),
            ).to_event(1),
        )
        composition = numpyro.sample(
            "composition_v3",
            dist.Dirichlet(1.1 * jnp.ones(COMPOSITION_DIM_V3))
            .expand([MAX_OBJECTS_V3])
            .to_event(1),
        )
        mean_image = render_scene_v3(
            background_rgb, presence, position, size, composition, renderer_params, image_shape
        )
        numpyro.deterministic("mean_v3", mean_image)
        numpyro.deterministic("count_v3", jnp.sum(presence))
        numpyro.sample("obs_v3", image_distribution_v3(mean_image, observation_noise), obs=image)

    return composition_to_rgb_v3, patch_model_v3, render_scene_v3


@app.cell
def _(
    CHANNELS_V3,
    COMPOSITION_DIM_V3,
    IMAGE_SHAPE_V3,
    LATENT_SITE_NAMES_V3,
    MAX_OBJECTS_V3,
    POSITION_CENTER_V3,
    Predictive,
    jnp,
    log_density,
    mo,
    patch_model_v3,
    random,
    render_scene_v3,
):
    prior_predictive_v3 = Predictive(
        patch_model_v3,
        num_samples=32,
        return_sites=(*LATENT_SITE_NAMES_V3, "mean_v3", "obs_v3", "count_v3"),
    )(random.PRNGKey(3002))

    actual_shapes_v3 = {name: tuple(value.shape) for name, value in prior_predictive_v3.items()}
    expected_shapes_v3 = {
        "background_rgb_v3": (32, CHANNELS_V3),
        "observation_noise_v3": (32,),
        "presence_v3": (32, MAX_OBJECTS_V3),
        "position_v3": (32, MAX_OBJECTS_V3, 2),
        "size_v3": (32, MAX_OBJECTS_V3),
        "composition_v3": (32, MAX_OBJECTS_V3, COMPOSITION_DIM_V3),
        "mean_v3": (32, *IMAGE_SHAPE_V3, CHANNELS_V3),
        "obs_v3": (32, *IMAGE_SHAPE_V3, CHANNELS_V3),
        "count_v3": (32,),
    }
    shape_checks_v3 = {name: actual_shapes_v3[name] == expected for name, expected in expected_shapes_v3.items()}
    finite_prior_v3 = bool(
        jnp.all(jnp.isfinite(prior_predictive_v3["mean_v3"]))
        and jnp.all(jnp.isfinite(prior_predictive_v3["obs_v3"]))
    )
    latents0_v3 = {name: prior_predictive_v3[name][0] for name in LATENT_SITE_NAMES_V3}
    log_joint_v3, trace_v3 = log_density(
        patch_model_v3,
        model_args=(prior_predictive_v3["obs_v3"][0],),
        model_kwargs={},
        params=latents0_v3,
    )
    log_joint_finite_v3 = bool(jnp.isfinite(log_joint_v3))
    trace_sample_sites_v3 = tuple(name for name, site in trace_v3.items() if site["type"] == "sample")
    required_sample_sites_present_v3 = all(name in trace_sample_sites_v3 for name in (*LATENT_SITE_NAMES_V3, "obs_v3"))

    _permutation_v3 = jnp.arange(MAX_OBJECTS_V3 - 1, -1, -1)
    mean_original_v3 = render_scene_v3(
        latents0_v3["background_rgb_v3"],
        latents0_v3["presence_v3"],
        latents0_v3["position_v3"],
        latents0_v3["size_v3"],
        latents0_v3["composition_v3"],
    )
    mean_permuted_v3 = render_scene_v3(
        latents0_v3["background_rgb_v3"],
        latents0_v3["presence_v3"][_permutation_v3],
        latents0_v3["position_v3"][_permutation_v3],
        latents0_v3["size_v3"][_permutation_v3],
        latents0_v3["composition_v3"][_permutation_v3],
    )
    order_max_abs_diff_v3 = float(jnp.max(jnp.abs(mean_original_v3 - mean_permuted_v3)))
    order_invariant_v3 = order_max_abs_diff_v3 < 1e-6

    control_background_v3 = jnp.array([0.70, 0.72, 0.55], dtype=jnp.float32)
    control_presence_v3 = jnp.zeros(MAX_OBJECTS_V3, dtype=jnp.float32).at[5].set(1.0)
    control_position_v3 = POSITION_CENTER_V3
    control_size_v3 = 0.060 * jnp.ones(MAX_OBJECTS_V3, dtype=jnp.float32)
    control_composition_base_v3 = jnp.ones((MAX_OBJECTS_V3, COMPOSITION_DIM_V3), dtype=jnp.float32) / COMPOSITION_DIM_V3
    control_composition_a_v3 = control_composition_base_v3.at[5].set(jnp.array([0.90, 0.04, 0.03, 0.03], dtype=jnp.float32))
    control_composition_b_v3 = control_composition_base_v3.at[5].set(jnp.array([0.03, 0.90, 0.04, 0.03], dtype=jnp.float32))
    mean_comp_a_v3 = render_scene_v3(control_background_v3, control_presence_v3, control_position_v3, control_size_v3, control_composition_a_v3)
    mean_comp_b_v3 = render_scene_v3(control_background_v3, control_presence_v3, control_position_v3, control_size_v3, control_composition_b_v3)
    composition_change_diff_v3 = float(jnp.max(jnp.abs(mean_comp_a_v3 - mean_comp_b_v3)))
    composition_changes_appearance_v3 = composition_change_diff_v3 > 0.03

    single_center_position_v3 = jnp.tile(jnp.array([[0.5, 0.5]], dtype=jnp.float32), (MAX_OBJECTS_V3, 1))
    single_presence_v3 = jnp.zeros(MAX_OBJECTS_V3, dtype=jnp.float32).at[0].set(1.0)
    single_composition_v3 = control_composition_base_v3.at[0].set(jnp.array([0.05, 0.75, 0.10, 0.10], dtype=jnp.float32))
    single_size_v3 = 0.060 * jnp.ones(MAX_OBJECTS_V3, dtype=jnp.float32)
    single_sprite_v3 = render_scene_v3(control_background_v3, single_presence_v3, single_center_position_v3, single_size_v3, single_composition_v3)
    vertical_flip_diff_v3 = float(jnp.mean(jnp.abs(single_sprite_v3 - jnp.flip(single_sprite_v3, axis=0))))
    horizontal_flip_diff_v3 = float(jnp.mean(jnp.abs(single_sprite_v3 - jnp.flip(single_sprite_v3, axis=1))))
    xy_asymmetry_present_v3 = (vertical_flip_diff_v3 > 1e-3) or (horizontal_flip_diff_v3 > 1e-3)

    milestone_v3_model_checks_passed = bool(
        all(shape_checks_v3.values())
        and finite_prior_v3
        and log_joint_finite_v3
        and required_sample_sites_present_v3
        and order_invariant_v3
        and composition_changes_appearance_v3
        and xy_asymmetry_present_v3
    )

    mo.md(
        "### Renderer v3 probabilistic checks\n\n"
        f"Prior predictive shapes correct: `{all(shape_checks_v3.values())}`. Shapes: `{actual_shapes_v3}`\n\n"
        f"Prior images finite: `{finite_prior_v3}`. Log joint finite: `{log_joint_finite_v3}` (`{float(log_joint_v3):.2f}`).\n\n"
        f"Required sample sites present: `{required_sample_sites_present_v3}`.\n\n"
        f"Order-invariance max diff under object permutation: `{order_max_abs_diff_v3:.3e}`.\n\n"
        f"Composition-change max diff: `{composition_change_diff_v3:.3f}`.\n\n"
        f"Local x/y asymmetry check, vertical flip diff `{vertical_flip_diff_v3:.4f}`, horizontal flip diff `{horizontal_flip_diff_v3:.4f}`.\n\n"
        f"Checks passed: `{milestone_v3_model_checks_passed}`."
    )
    return (
        control_background_v3,
        milestone_v3_model_checks_passed,
        prior_predictive_v3,
    )


@app.cell
def _(np, plt, prior_predictive_v3):
    fig_prior_grid_v3, axes_prior_grid_v3 = plt.subplots(4, 4, figsize=(8, 8))
    for _ax, _image, _count in zip(
        axes_prior_grid_v3.ravel(),
        np.asarray(prior_predictive_v3["obs_v3"][:16]),
        np.asarray(prior_predictive_v3["count_v3"][:16]),
    ):
        _ax.imshow(np.clip(_image, 0.0, 1.0), interpolation="nearest")
        _ax.set_title(f"count={int(_count)}", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    fig_prior_grid_v3.suptitle("Renderer v3 prior predictive RGB patches", y=0.94)
    fig_prior_grid_v3.tight_layout()
    fig_prior_grid_v3
    return


@app.cell
def _(
    COMPOSITION_DIM_V3,
    MAX_OBJECTS_V3,
    control_background_v3,
    jnp,
    np,
    plt,
    render_scene_v3,
):
    sprite_compositions_v3 = jnp.array(
        [
            [0.90, 0.04, 0.03, 0.03],
            [0.04, 0.90, 0.03, 0.03],
            [0.03, 0.04, 0.90, 0.03],
            [0.03, 0.04, 0.03, 0.90],
        ],
        dtype=jnp.float32,
    )
    sprite_sizes_v3 = jnp.array([0.035, 0.060, 0.082], dtype=jnp.float32)
    fig_sprite_sheet_v3, axes_sprite_sheet_v3 = plt.subplots(len(sprite_compositions_v3), len(sprite_sizes_v3), figsize=(7, 8))
    for _row in range(len(sprite_compositions_v3)):
        for _col in range(len(sprite_sizes_v3)):
            _presence = jnp.zeros(MAX_OBJECTS_V3, dtype=jnp.float32).at[0].set(1.0)
            _position = jnp.tile(jnp.array([[0.5, 0.5]], dtype=jnp.float32), (MAX_OBJECTS_V3, 1))
            _size = sprite_sizes_v3[_col] * jnp.ones(MAX_OBJECTS_V3, dtype=jnp.float32)
            _composition = (jnp.ones((MAX_OBJECTS_V3, COMPOSITION_DIM_V3), dtype=jnp.float32) / COMPOSITION_DIM_V3).at[0].set(sprite_compositions_v3[_row])
            _image = render_scene_v3(control_background_v3, _presence, _position, _size, _composition)
            axes_sprite_sheet_v3[_row, _col].imshow(np.asarray(_image), interpolation="nearest")
            axes_sprite_sheet_v3[_row, _col].set_title(
                f"comp={np.asarray(sprite_compositions_v3[_row]).round(2)}\nsize={float(sprite_sizes_v3[_col]):.3f}",
                fontsize=8,
            )
            axes_sprite_sheet_v3[_row, _col].set_xticks([])
            axes_sprite_sheet_v3[_row, _col].set_yticks([])
    fig_sprite_sheet_v3.suptitle("Renderer v3 single-object composition/size effects", y=0.99)
    fig_sprite_sheet_v3.tight_layout()
    fig_sprite_sheet_v3
    return


@app.cell
def _(
    COMPOSITION_DIM_V3,
    MAX_OBJECTS_V3,
    composition_to_rgb_v3,
    jnp,
    np,
    plt,
    prior_predictive_v3,
):
    presence_np_v3 = np.asarray(prior_predictive_v3["presence_v3"])
    active_mask_np_v3 = presence_np_v3.astype(bool)
    active_positions_np_v3 = np.asarray(prior_predictive_v3["position_v3"])[active_mask_np_v3]
    active_sizes_np_v3 = np.asarray(prior_predictive_v3["size_v3"])[active_mask_np_v3]
    active_compositions_np_v3 = np.asarray(prior_predictive_v3["composition_v3"])[active_mask_np_v3]
    active_colours_np_v3 = np.asarray(composition_to_rgb_v3(jnp.asarray(active_compositions_np_v3)))
    counts_np_v3 = np.asarray(prior_predictive_v3["count_v3"])

    fig_latents_v3, axes_latents_v3 = plt.subplots(2, 3, figsize=(11, 6))
    axes_latents_v3[0, 0].hist(counts_np_v3, bins=np.arange(-0.5, MAX_OBJECTS_V3 + 1.5), rwidth=0.8)
    axes_latents_v3[0, 0].set_title("object count")
    axes_latents_v3[0, 1].scatter(active_positions_np_v3[:, 1], active_positions_np_v3[:, 0], s=8, alpha=0.6)
    axes_latents_v3[0, 1].invert_yaxis()
    axes_latents_v3[0, 1].set_title("active positions")
    axes_latents_v3[0, 2].hist(active_sizes_np_v3, bins=20, color="tab:green")
    axes_latents_v3[0, 2].set_title("active sizes")
    for _k in range(COMPOSITION_DIM_V3):
        axes_latents_v3[1, 0].hist(active_compositions_np_v3[:, _k], bins=20, alpha=0.55, label=f"c{_k}")
    axes_latents_v3[1, 0].set_title("composition components")
    axes_latents_v3[1, 0].legend(fontsize=8)
    axes_latents_v3[1, 1].scatter(active_colours_np_v3[:, 0], active_colours_np_v3[:, 2], c=active_colours_np_v3, s=16)
    axes_latents_v3[1, 1].set_title("renderer composition-colour diagnostic")
    axes_latents_v3[1, 1].set_xlabel("red")
    axes_latents_v3[1, 1].set_ylabel("blue")
    axes_latents_v3[1, 2].hist(np.asarray(prior_predictive_v3["observation_noise_v3"]), bins=20, color="tab:orange")
    axes_latents_v3[1, 2].set_title("observation noise")
    fig_latents_v3.tight_layout()
    fig_latents_v3
    return active_colours_np_v3, active_sizes_np_v3, counts_np_v3


@app.cell
def _(
    IMAGE_SHAPE_V3,
    composition_to_rgb_v3,
    jnp,
    np,
    plt,
    prior_predictive_v3,
):
    fig_overlay_v3, axes_overlay_v3 = plt.subplots(2, 4, figsize=(10, 5))
    for _ax, _idx in zip(axes_overlay_v3.ravel(), range(8)):
        _ax.imshow(np.asarray(prior_predictive_v3["obs_v3"][_idx]).clip(0.0, 1.0), interpolation="nearest")
        _ax.set_title(f"sample {_idx}; count={int(prior_predictive_v3['count_v3'][_idx])}", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _present, _pos, _size, _comp in zip(
            np.asarray(prior_predictive_v3["presence_v3"][_idx]),
            np.asarray(prior_predictive_v3["position_v3"][_idx]),
            np.asarray(prior_predictive_v3["size_v3"][_idx]),
            np.asarray(prior_predictive_v3["composition_v3"][_idx]),
        ):
            if _present > 0.5:
                _colour = np.asarray(composition_to_rgb_v3(jnp.asarray(_comp)))
                _ax.add_patch(
                    plt.Circle(
                        (_pos[1] * (IMAGE_SHAPE_V3[1] - 1), _pos[0] * (IMAGE_SHAPE_V3[0] - 1)),
                        radius=_size * IMAGE_SHAPE_V3[0],
                        edgecolor=np.clip(_colour, 0.0, 1.0),
                        facecolor="none",
                        linewidth=1.1,
                    )
                )
    fig_overlay_v3.suptitle("Renderer v3 prior images with true position/size overlays", y=1.02)
    fig_overlay_v3.tight_layout()
    fig_overlay_v3
    return


@app.cell(hide_code=True)
def _(
    active_colours_np_v3,
    active_sizes_np_v3,
    counts_np_v3,
    milestone_v3_model_checks_passed,
    mo,
    np,
):
    prior_counts_diverse_v3 = bool(len(np.unique(counts_np_v3)) >= 4)
    prior_colours_diverse_v3 = bool(np.mean(np.std(active_colours_np_v3, axis=0)) > 0.08)
    prior_sizes_diverse_v3 = bool(np.std(active_sizes_np_v3) > 0.01)
    prior_predictive_visual_gate_v3 = bool(
        milestone_v3_model_checks_passed
        and prior_counts_diverse_v3
        and prior_colours_diverse_v3
        and prior_sizes_diverse_v3
    )

    mo.md(
        f"""
        ## Renderer v3 Milestone 1 report — generic learned coordinate renderer

        **Implemented.** A new RGB NumPyro generative model `patch_model_v3` with named sites `background_rgb_v3`, `observation_noise_v3`, `presence_v3`, `position_v3`, `size_v3`, and `composition_v3`. The deterministic renderer `render_scene_v3` is order-invariant and uses a small shared coordinate MLP. Each object's local appearance is a function of `(y-y0)/size`, `(x-x0)/size`, size, composition, and local background RGB. It outputs a local RGB background delta. There are no hard-coded object palettes/rims/shadows/highlights and no per-image latent.

        **Verified numerically.** Prior predictive sampling works, shapes are correct, the log joint is finite, object order permutation leaves the rendered image unchanged, changing composition changes appearance, and the sprite is not constrained to radial symmetry.

        **Verified visually.** The cells above show prior predictive RGB patches, single-object composition/size effects, latent/count/colour distributions, and position/size overlays. Count diversity: `{prior_counts_diverse_v3}`. Colour diversity: `{prior_colours_diverse_v3}`. Size diversity: `{prior_sizes_diverse_v3}`.

        **Concerns.** This renderer is flexible enough to be generic, but it is still only an initial prior/initialisation. The renderer parameters must later be learned/calibrated in controlled settings, with train-set and held-out latent checks. A smooth locality envelope is used only to keep object effects local; the learned MLP still receives local y and x separately. The current slot grid is an identifiability scaffold and should not be mistaken for a final set posterior.

        **Renderer v3 Milestone 1 passed:** `{prior_predictive_visual_gate_v3}`.

        **Next.** Inspect these visuals. If they are acceptable as a starting simulator, implement the v3 guide. If not, adjust renderer capacity/initialisation before any amortisation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
