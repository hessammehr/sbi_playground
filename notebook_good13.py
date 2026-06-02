# /// script
# dependencies = [
#     "flax==0.12.7",
#     "jax[cuda12]==0.10.1",
#     "jaxlib==0.10.1",
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "numpyro==0.21.0",
#     "optax==0.2.8",
#     "pillow==12.2.0",
#     "tqdm==4.67.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Memory-safe Flax/NumPyro SBI for microscopy droplet patches

    This is a systematic replacement for the previous condensed notebook. The active
    simulator is the expressive v4 object renderer from `notebook4.py`: explicit
    interpretable latents (presence/count, position, size, composition, background) and
    a global/shared learned coordinate renderer. The amortised posterior is a **NumPyro
    guide** whose neural network is a **Flax Linen module** registered with NumPyro via
    `numpyro.contrib.module.flax_module`.

    The memory issue was not the problem size; it was materialising full
    `batch × objects × pixels × hidden` tensors. Rendering is now object-chunked, and
    all synthetic generation / log-joint evaluation is chunked. Script mode defaults to
    a small smoke profile so `uv run notebook5.py` is a safe OOM check. Larger runs are
    selected with the runtime-profile widget.
    """)
    return


@app.cell
def _():
    import os

    # Must be set before importing JAX. On GPU/Metal this avoids allocator surprises;
    # on CPU it is harmless. The renderer itself is chunked, so no preallocation is needed.
    # os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")


    import gc
    import math
    import resource
    import sys
    import time
    from dataclasses import dataclass, field
    from typing import Callable

    import numpy as np
    import matplotlib.pyplot as plt

    import jax
    import jax.numpy as jnp
    import jax.nn as jnn
    from jax import random

    import flax.linen as flax_nn

    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from numpyro.contrib.module import flax_module
    from numpyro.infer import Predictive

    import optax
    from tqdm.auto import tqdm

    numpyro.set_host_device_count(1)
    return (
        Callable,
        Predictive,
        dataclass,
        dist,
        field,
        flax_module,
        flax_nn,
        gc,
        handlers,
        jax,
        jnn,
        jnp,
        np,
        numpyro,
        optax,
        plt,
        random,
        resource,
        sys,
        time,
        tqdm,
    )


@app.cell
def _(jax):
    jax.devices()
    return


@app.cell
def _(dataclass, field, gc, jax, resource, sys, time):
    @dataclass
    class MemoryLog:
        records: list = field(default_factory=list)

        def mark(self, label):
            gc.collect()
            self.records.append((label, round(max_rss_mb(), 1), round(time.time(), 1)))
            return self.records[-1]

    def max_rss_mb():
        """Maximum resident set size in MiB (macOS reports bytes; Linux reports KiB)."""
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024

    def block_until_ready_tree(tree):
        for leaf in jax.tree_util.tree_leaves(tree):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        return tree

    return MemoryLog, block_until_ready_tree


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class RunConfig:
        profile: str
        run_training: bool
        train_size: int
        val_size: int
        train_steps: int
        batch_size: int
        learning_rate: float
        eval_every: int
        prior_visual_samples: int
        simulate_chunk: int
        eval_chunk: int
        run_sc: bool
        sc_patches: int
        sc_steps: int
        sc_proposals: int
        sc_fixed_proposals: int

    profile_defaults = {
        # The default for `uv run notebook5.py`: verifies simulation, gradients,
        # training loop, metrics, and visuals without threatening memory.
        "smoke": dict(train_size=8, val_size=4, train_steps=3, batch_size=2,
                      prior_visual_samples=8, simulate_chunk=4, eval_chunk=8,
                      sc_patches=1, sc_steps=0, sc_proposals=2, sc_fixed_proposals=2),
        # Useful enough for qualitative diagnostics on a laptop while still bounded.
        "useful": dict(train_size=128, val_size=32, train_steps=200, batch_size=4,
                       prior_visual_samples=8, simulate_chunk=4, eval_chunk=32,
                       sc_patches=4, sc_steps=20, sc_proposals=4, sc_fixed_proposals=6),
        # Still chunked; intended for overnight/desktop runs.
        "full": dict(train_size=4096, val_size=128, train_steps=5000, batch_size=128,
                     prior_visual_samples=8, simulate_chunk=128, eval_chunk=128,
                     sc_patches=32, sc_steps=50, sc_proposals=6, sc_fixed_proposals=10),
    }
    return RunConfig, profile_defaults


@app.cell
def _(mo, profile_defaults):
    profile_dropdown = mo.ui.dropdown(
        options=list(profile_defaults.keys()),
        value="full",
        label="Runtime profile",
    )
    mo.vstack([
        mo.md("### Runtime profile"),
        profile_dropdown,
        mo.md("Changing the profile recreates the numeric controls below with that profile's defaults."),
    ])
    return (profile_dropdown,)


@app.cell
def _(mo, profile_defaults, profile_dropdown):
    _defaults = profile_defaults.get(profile_dropdown.value, profile_defaults["smoke"])
    run_training_widget = mo.ui.switch(value=True, label="Run NPE training")
    train_size_widget = mo.ui.number(start=1, stop=100000, step=1, value=_defaults["train_size"], label="train size")
    val_size_widget = mo.ui.number(start=1, stop=100000, step=1, value=_defaults["val_size"], label="validation size")
    train_steps_widget = mo.ui.number(start=0, stop=100000, step=1, value=_defaults["train_steps"], label="training steps")
    batch_size_widget = mo.ui.number(start=1, stop=4096, step=1, value=_defaults["batch_size"], label="batch size")
    learning_rate_widget = mo.ui.number(start=1e-6, stop=1e-1, step=1e-4, value=5e-4, label="learning rate")
    eval_every_widget = mo.ui.number(start=1, stop=100000, step=1, value=max(1, min(50, _defaults["train_steps"])), label="eval every")
    prior_visual_samples_widget = mo.ui.number(start=1, stop=10000, step=1, value=_defaults["prior_visual_samples"], label="prior visual samples")
    simulate_chunk_widget = mo.ui.number(start=1, stop=256, step=1, value=_defaults["simulate_chunk"], label="simulation chunk")
    eval_chunk_widget = mo.ui.number(start=1, stop=4096, step=1, value=_defaults["eval_chunk"], label="eval chunk")
    run_sc_widget = mo.ui.switch(value=False, label="Run real-image SC")
    sc_patches_widget = mo.ui.number(start=1, stop=64, step=1, value=_defaults["sc_patches"], label="SC real patches")
    sc_steps_widget = mo.ui.number(start=0, stop=10000, step=1, value=_defaults["sc_steps"], label="SC steps")
    sc_proposals_widget = mo.ui.number(start=1, stop=128, step=1, value=_defaults["sc_proposals"], label="SC proposals")
    sc_fixed_proposals_widget = mo.ui.number(start=1, stop=128, step=1, value=_defaults["sc_fixed_proposals"], label="SC fixed proposals")

    mo.vstack([
        mo.md("### Run controls"),
        mo.hstack([run_training_widget, run_sc_widget]),
        mo.hstack([train_size_widget, val_size_widget, train_steps_widget, batch_size_widget]),
        mo.hstack([learning_rate_widget, eval_every_widget, prior_visual_samples_widget]),
        mo.hstack([simulate_chunk_widget, eval_chunk_widget]),
        mo.hstack([sc_patches_widget, sc_steps_widget, sc_proposals_widget, sc_fixed_proposals_widget]),
    ])
    return (
        batch_size_widget,
        eval_chunk_widget,
        eval_every_widget,
        learning_rate_widget,
        prior_visual_samples_widget,
        run_sc_widget,
        run_training_widget,
        sc_fixed_proposals_widget,
        sc_patches_widget,
        sc_proposals_widget,
        sc_steps_widget,
        simulate_chunk_widget,
        train_size_widget,
        train_steps_widget,
        val_size_widget,
    )


@app.cell
def _(
    RunConfig,
    batch_size_widget,
    eval_chunk_widget,
    eval_every_widget,
    learning_rate_widget,
    mo,
    prior_visual_samples_widget,
    profile_defaults,
    profile_dropdown,
    run_sc_widget,
    run_training_widget,
    sc_fixed_proposals_widget,
    sc_patches_widget,
    sc_proposals_widget,
    sc_steps_widget,
    simulate_chunk_widget,
    train_size_widget,
    train_steps_widget,
    val_size_widget,
):
    _defaults = profile_defaults.get(profile_dropdown.value, profile_defaults["smoke"])

    def _int_value(widget, name):
        return int(_defaults[name] if widget.value is None else widget.value)

    def _float_value(widget, default):
        return float(default if widget.value is None else widget.value)

    run_config = RunConfig(
        profile=profile_dropdown.value,
        run_training=bool(run_training_widget.value),
        train_size=_int_value(train_size_widget, "train_size"),
        val_size=_int_value(val_size_widget, "val_size"),
        train_steps=_int_value(train_steps_widget, "train_steps"),
        batch_size=_int_value(batch_size_widget, "batch_size"),
        learning_rate=_float_value(learning_rate_widget, 5e-4),
        eval_every=int(max(1, eval_every_widget.value or max(1, min(50, _defaults["train_steps"])))),
        prior_visual_samples=_int_value(prior_visual_samples_widget, "prior_visual_samples"),
        simulate_chunk=_int_value(simulate_chunk_widget, "simulate_chunk"),
        eval_chunk=_int_value(eval_chunk_widget, "eval_chunk"),
        run_sc=bool(run_sc_widget.value),
        sc_patches=_int_value(sc_patches_widget, "sc_patches"),
        sc_steps=_int_value(sc_steps_widget, "sc_steps"),
        sc_proposals=_int_value(sc_proposals_widget, "sc_proposals"),
        sc_fixed_proposals=_int_value(sc_fixed_proposals_widget, "sc_fixed_proposals"),
    )

    mo.md(
        f"**Runtime profile:** `{run_config.profile}` | "
        f"train/val `{run_config.train_size}/{run_config.val_size}`, "
        f"steps `{run_config.train_steps}`, batch `{run_config.batch_size}`, "
        f"simulation chunk `{run_config.simulate_chunk}`, SC enabled `{run_config.run_sc}`."
    )
    return (run_config,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Shared Flax modules

    `SlotGuideNet` is the slot-aligned posterior encoder from `notebook4.py`, rewritten
    as a Flax module. It maps a 64×64 RGB image to per-cell posterior parameters plus
    global background/noise/illumination parameters. The NumPyro guide below instantiates
    it through `flax_module`, so the neural network is registered as NumPyro parameters
    at the guide site `v4_guide$params`.

    `FourierObjectRenderer` is the v4 learned local renderer, also rewritten as Flax.
    Its parameters are global/shared fixed simulator parameters in this notebook; no
    per-image latent or object embedding is introduced.
    """)
    return


@app.cell
def _(flax_nn, jnp, np, random):
    def softplus_inverse(y):
        y = np.asarray(y, dtype=np.float32)
        return np.log(np.expm1(y)).astype(np.float32)

    def constant_bias(values):
        values = tuple(float(v) for v in np.asarray(values, dtype=np.float32).ravel())

        def init(key, shape, dtype=jnp.float32):
            del key
            return jnp.broadcast_to(jnp.asarray(values, dtype), shape)

        return init

    class SlotGuideNet(flax_nn.Module):
        grid: int
        slot_out: int
        global_out: int
        slot_bias: tuple
        global_bias: tuple
        hidden: int = 128
        conv_channels: tuple = (24, 48, 64)

        @flax_nn.compact
        def __call__(self, image_batch):
            x = image_batch.astype(jnp.float32)
            x = flax_nn.relu(
                flax_nn.Conv(self.conv_channels[0], (5, 5), strides=(2, 2), padding="SAME", name="conv1")(x)
            )
            x = flax_nn.relu(
                flax_nn.Conv(self.conv_channels[1], (3, 3), strides=(2, 2), padding="SAME", name="conv2")(x)
            )
            x = flax_nn.relu(
                flax_nn.Conv(self.conv_channels[2], (3, 3), strides=(2, 2), padding="SAME", name="conv3")(x)
            )
            batch, height, width, depth = x.shape
            if height == self.grid and width == self.grid:
                slot_features = x.reshape(batch, self.grid * self.grid, depth)
            else:
                block_y = height // self.grid
                block_x = width // self.grid
                slot_features = (
                    x.reshape(batch, self.grid, block_y, self.grid, block_x, depth)
                    .mean(axis=(2, 4))
                    .reshape(batch, self.grid * self.grid, depth)
                )
            global_feature = x.mean(axis=(1, 2))
            tiled_global = jnp.broadcast_to(global_feature[:, None, :], slot_features.shape)
            slot_input = jnp.concatenate([slot_features, tiled_global], axis=-1)
            h = flax_nn.relu(flax_nn.Dense(self.hidden, name="slot_dense1")(slot_input))
            h = flax_nn.relu(flax_nn.Dense(self.hidden, name="slot_dense2")(h))
            per_slot = flax_nn.Dense(
                self.slot_out,
                kernel_init=flax_nn.initializers.normal(0.02),
                bias_init=constant_bias(self.slot_bias),
                name="slot_out",
            )(h)
            global_out = flax_nn.Dense(
                self.global_out,
                kernel_init=flax_nn.initializers.normal(0.02),
                bias_init=constant_bias(self.global_bias),
                name="global_out",
            )(global_feature)
            return per_slot, global_out

    class FourierObjectRenderer(flax_nn.Module):
        composition_dim: int
        channels: int = 3
        hidden: int = 48
        n_fourier: int = 6
        locality_sigma: float = 0.8

        @flax_nn.compact
        def __call__(self, local_yx, size, composition, local_background):
            freqs = self.param(
                "fourier_freqs",
                lambda key, shape: random.normal(key, shape, dtype=jnp.float32),
                (2, self.n_fourier),
            )
            proj = jnp.einsum("ohwk,kf->ohwf", local_yx, freqs)
            fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
            size_feature = jnp.broadcast_to(
                jnp.log(size[:, None, None, None] / 0.1), local_yx.shape[:-1] + (1,)
            )
            comp_feature = jnp.broadcast_to(
                composition[:, None, None, :], local_yx.shape[:-1] + (self.composition_dim,)
            )
            features = jnp.concatenate([fourier, size_feature, comp_feature, local_background], axis=-1)

            def first_kernel_init(key, shape, dtype=jnp.float32):
                fan_in = shape[0]
                weights = random.normal(key, shape, dtype=dtype) / jnp.sqrt(fan_in)
                comp_start = 2 * self.n_fourier + 1
                return weights.at[comp_start:comp_start + self.composition_dim, :].multiply(7.0)

            z = jnp.tanh(
                flax_nn.Dense(self.hidden, kernel_init=first_kernel_init, name="dense1")(features)
            )
            z = jnp.tanh(flax_nn.Dense(self.hidden, name="dense2")(z))
            out = flax_nn.Dense(
                1 + self.channels,
                kernel_init=flax_nn.initializers.normal(3.5 / np.sqrt(self.hidden)),
                bias_init=constant_bias([1.2, 0.0, 0.0, 0.0]),
                name="dense_out",
            )(z)
            locality = jnp.exp(-0.5 * jnp.sum((local_yx / self.locality_sigma) ** 2, axis=-1))
            alpha = locality * flax_nn.sigmoid(out[..., 0])
            rgb = flax_nn.sigmoid(out[..., 1:])
            return alpha, rgb

    return FourierObjectRenderer, SlotGuideNet, softplus_inverse


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
        site_names: tuple
        position_low: object
        position_high: object
        size_low: float
        size_high: float
        model: Callable
        init_guide_params: Callable
        guide: Callable
        guide_log_prob: Callable
        guide_point_estimates: Callable
        render: Callable
        model_log_joint: Callable
        composition_to_rgb: Callable
        predictive_sites: tuple
        extras: dict = field(default_factory=dict)

    return (ModelVersion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Active v4 SBI model

    The object renderer is chunked over object cells (`object_chunk=8`), so the largest
    hidden activation scales like `batch_chunk × object_chunk × H × W × hidden` rather
    than `batch × 64 × H × W × hidden`. Synthetic generation and SC log-joints use small
    outer chunks on top of this.
    """)
    return


@app.cell
def _(
    FourierObjectRenderer,
    ModelVersion,
    SlotGuideNet,
    dist,
    flax_module,
    handlers,
    jax,
    jnn,
    jnp,
    np,
    numpyro,
    random,
    softplus_inverse,
):
    def build_v4(renderer_init=None):
        image_shape = (64, 64)
        channels = 3
        grid_size = 8
        max_objects = grid_size * grid_size
        composition_dim = 3
        size_low, size_high = 0.02, 0.40
        presence_prob = 7.0 / max_objects
        object_chunk = 8
        assert max_objects % object_chunk == 0

        cell_y = jnp.arange(grid_size, dtype=jnp.float32) / grid_size
        cell_x = jnp.arange(grid_size, dtype=jnp.float32) / grid_size
        cy, cx = jnp.meshgrid(cell_y, cell_x, indexing="ij")
        position_low = jnp.stack([cy.ravel(), cx.ravel()], axis=-1)
        position_high = position_low + 1.0 / grid_size
        position_scale = position_high - position_low
        position_center = 0.5 * (position_low + position_high)
        bg_low = jnp.array([0.40, 0.40, 0.35], dtype=jnp.float32)
        bg_high = jnp.array([0.80, 0.80, 0.75], dtype=jnp.float32)
        sites = (
            "background_rgb",
            "bg_gradient",
            "observation_noise",
            "presence",
            "position",
            "size",
            "composition",
        )

        y = jnp.linspace(0.0, 1.0, image_shape[0], dtype=jnp.float32)
        x = jnp.linspace(0.0, 1.0, image_shape[1], dtype=jnp.float32)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        pixel_grid = jnp.stack([yy, xx], axis=-1)

        renderer_module = FourierObjectRenderer(
            composition_dim=composition_dim,
            channels=channels,
            hidden=32,
            n_fourier=6,
            locality_sigma=0.8,
        )
        if renderer_init is None:
            initial_renderer_params = renderer_module.init(
                random.PRNGKey(4001),
                jnp.zeros((object_chunk, *image_shape, 2), dtype=jnp.float32),
                0.12 * jnp.ones((object_chunk,), dtype=jnp.float32),
                jnp.ones((object_chunk, composition_dim), dtype=jnp.float32) / composition_dim,
                0.6 * jnp.ones((object_chunk, *image_shape, channels), dtype=jnp.float32),
            )["params"]
        else:
            initial_renderer_params = renderer_init

        # Name of the NumPyro param site that holds the renderer's Flax params.
        # Registered inside `model(...)` so SVI/score-based training can see them;
        # NPE never touches model params, so this is gradient-free under the current
        # training loop -- it is plumbing for Step 2.
        renderer_param_name = "v4_renderer$params"

        def background_field(background_rgb, bg_gradient):
            centred = pixel_grid - 0.5
            return jnp.clip(
                background_rgb[None, None, :] + jnp.einsum("hwk,kc->hwc", centred, bg_gradient),
                0.0,
                1.0,
            )

        def render_scene(background_rgb, bg_gradient, presence, position, size, composition, renderer_params):
            background = background_field(background_rgb, bg_gradient)
            n_chunks = max_objects // object_chunk
            positions = position.reshape(n_chunks, object_chunk, 2)
            sizes = size.reshape(n_chunks, object_chunk)
            comps = composition.reshape(n_chunks, object_chunk, composition_dim)
            presences = presence.reshape(n_chunks, object_chunk)

            def body(carry, xs):
                total_alpha, weighted_rgb = carry
                pos_c, size_c, comp_c, pres_c = xs
                local = (pixel_grid[None, :, :, :] - pos_c[:, None, None, :]) / (
                    size_c[:, None, None, None] + 1e-6
                )
                local_bg = jnp.broadcast_to(background[None, :, :, :], (object_chunk, *image_shape, channels))
                alpha, rgb = renderer_module.apply({"params": renderer_params}, local, size_c, comp_c, local_bg)
                effective_alpha = pres_c[:, None, None] * alpha
                return (
                    total_alpha + jnp.sum(effective_alpha, axis=0),
                    weighted_rgb + jnp.sum(effective_alpha[..., None] * rgb, axis=0),
                ), None

            init = (
                jnp.zeros(image_shape, dtype=jnp.float32),
                jnp.zeros((*image_shape, channels), dtype=jnp.float32),
            )
            (total_alpha, weighted_rgb), _ = jax.lax.scan(
                body, init, (positions, sizes, comps, presences)
            )
            blend = weighted_rgb / (total_alpha[..., None] + 1e-6)
            coverage = 1.0 - jnp.exp(-total_alpha)
            return jnp.clip((1.0 - coverage[..., None]) * background + coverage[..., None] * blend, 0.0, 1.0)

        def size_prior_dist():
            return dist.TransformedDistribution(
                dist.Beta(1.3 * jnp.ones(max_objects), 3.0 * jnp.ones(max_objects)),
                dist.transforms.AffineTransform(size_low, size_high - size_low),
            ).to_event(1)

        def model(image=None):
            # Register the renderer Flax params as a NumPyro param site, initialised
            # from the deterministic PRNGKey(4001) draw. Behaviour is identical to
            # before -- the only effect is that these params are now discoverable in
            # the model's trace, ready for an ELBO-style loss to pick them up.
            renderer_params = numpyro.param(renderer_param_name, initial_renderer_params)
            background_rgb = numpyro.sample("background_rgb", dist.Uniform(bg_low, bg_high).to_event(1))
            bg_gradient = numpyro.sample("bg_gradient", dist.Normal(0.0, 0.08).expand([2, 3]).to_event(2))
            observation_noise = numpyro.sample("observation_noise", dist.LogNormal(jnp.log(0.02), 0.3))
            presence = numpyro.sample(
                "presence", dist.Bernoulli(presence_prob).expand([max_objects]).to_event(1)
            )
            position = numpyro.sample("position", dist.Uniform(position_low, position_high).to_event(2))
            size = numpyro.sample("size", size_prior_dist())
            composition = numpyro.sample(
                "composition", dist.Dirichlet(jnp.ones(composition_dim)).expand([max_objects]).to_event(1)
            )
            mean = render_scene(background_rgb, bg_gradient, presence, position, size, composition, renderer_params)
            numpyro.deterministic("mean", mean)
            numpyro.deterministic("count", jnp.sum(presence))
            numpyro.sample("obs", dist.Normal(mean, observation_noise).to_event(3), obs=image)

        def batched_model(images):
            """Batched counterpart of `model`, for SVI on multiple images at once.

            Mirrors `model` exactly except every sample site is inside a
            `numpyro.plate("batch", N)`. The renderer param site is registered
            ONCE (it is shared across the batch). Uses `render_latent_batch` so
            the per-image render is vmapped.
            """
            renderer_params = numpyro.param(renderer_param_name, initial_renderer_params)
            N = images.shape[0]
            with numpyro.plate("batch", N):
                background_rgb = numpyro.sample(
                    "background_rgb", dist.Uniform(bg_low, bg_high).to_event(1)
                )
                bg_gradient = numpyro.sample(
                    "bg_gradient", dist.Normal(0.0, 0.08).expand([2, 3]).to_event(2)
                )
                observation_noise = numpyro.sample(
                    "observation_noise", dist.LogNormal(jnp.log(0.02), 0.3)
                )
                presence = numpyro.sample(
                    "presence", dist.Bernoulli(presence_prob).expand([max_objects]).to_event(1)
                )
                position = numpyro.sample(
                    "position", dist.Uniform(position_low, position_high).to_event(2)
                )
                size = numpyro.sample("size", size_prior_dist())
                composition = numpyro.sample(
                    "composition", dist.Dirichlet(jnp.ones(composition_dim)).expand([max_objects]).to_event(1)
                )
                mean = render_latent_batch(
                    {
                        "background_rgb": background_rgb,
                        "bg_gradient": bg_gradient,
                        "presence": presence.astype(jnp.float32),
                        "position": position,
                        "size": size,
                        "composition": composition,
                    },
                    renderer_params,
                )
                numpyro.sample(
                    "obs",
                    dist.Normal(mean, observation_noise[:, None, None, None]).to_event(3),
                    obs=images,
                )

        def render_latent_batch(latents, renderer_params):
            return jax.vmap(lambda *args: render_scene(*args, renderer_params))(
                latents["background_rgb"],
                latents["bg_gradient"],
                latents["presence"],
                latents["position"],
                latents["size"],
                latents["composition"],
            )

        def model_log_joint(images, latents, renderer_params=None):
            if renderer_params is None:
                renderer_params = initial_renderer_params
            mean = render_latent_batch(latents, renderer_params)
            log_prob = dist.Uniform(bg_low, bg_high).to_event(1).log_prob(latents["background_rgb"])
            log_prob += dist.Normal(0.0, 0.08).expand([2, 3]).to_event(2).log_prob(latents["bg_gradient"])
            log_prob += dist.LogNormal(jnp.log(0.02), 0.3).log_prob(latents["observation_noise"])
            log_prob += dist.Bernoulli(presence_prob).expand([max_objects]).to_event(1).log_prob(latents["presence"])
            log_prob += dist.Uniform(position_low, position_high).to_event(2).log_prob(latents["position"])
            log_prob += size_prior_dist().log_prob(latents["size"])
            log_prob += dist.Dirichlet(jnp.ones(composition_dim)).expand([max_objects]).to_event(1).log_prob(latents["composition"])
            log_prob += dist.Normal(mean, latents["observation_noise"][:, None, None, None]).to_event(3).log_prob(images)
            return log_prob

        guide_slot_out = 1 + 4 + 2 + composition_dim
        guide_global_out = 2 * channels + 2 + 6
        slot_bias = np.concatenate([
            np.array([np.log(presence_prob / (1.0 - presence_prob))], dtype=np.float32),
            np.full((4,), softplus_inverse(2.0), dtype=np.float32),
            np.full((2,), softplus_inverse(2.0), dtype=np.float32),
            np.full((composition_dim,), softplus_inverse(1.0), dtype=np.float32),
        ])
        global_bias = np.concatenate([
            np.full((2 * channels,), softplus_inverse(4.0), dtype=np.float32),
            np.array([np.log(0.02), softplus_inverse(0.30)], dtype=np.float32),
            np.zeros((6,), dtype=np.float32),
        ])
        guide_module = SlotGuideNet(
            grid=grid_size,
            slot_out=guide_slot_out,
            global_out=guide_global_out,
            slot_bias=tuple(slot_bias.tolist()),
            global_bias=tuple(global_bias.tolist()),
            hidden=96,
            conv_channels=(16, 32, 48),
        )
        guide_param_name = "v4_guide$params"

        def ensure_image_batch(image):
            image = jnp.asarray(image, dtype=jnp.float32)
            if image.ndim == 3:
                return image[None, ...], True
            return image, False

        def positive(raw, floor=1e-3):
            return jnn.softplus(raw) + floor

        def raw_guide_outputs(guide_params, image):
            image_batch, _ = ensure_image_batch(image)
            return guide_module.apply({"params": guide_params}, image_batch)

        def parse_raw(per_slot, global_out):
            pos_raw = per_slot[..., 1:5].reshape(-1, max_objects, 2, 2)
            size_raw = per_slot[..., 5:7]
            comp_raw = per_slot[..., 7:7 + composition_dim]
            bg_raw = global_out[:, :2 * channels].reshape(-1, channels, 2)
            noise_raw = global_out[:, 2 * channels:2 * channels + 2]
            grad_raw = global_out[:, 2 * channels + 2:].reshape(-1, 2, 3)
            return {
                "background_alpha": positive(bg_raw[..., 0]),
                "background_beta": positive(bg_raw[..., 1]),
                "noise_loc": noise_raw[:, 0],
                "noise_scale": positive(noise_raw[:, 1], 0.02),
                "grad_loc": grad_raw,
                "presence_logits": per_slot[..., 0],
                "position_alpha": positive(pos_raw[..., 0]),
                "position_beta": positive(pos_raw[..., 1]),
                "size_alpha": positive(size_raw[..., 0]),
                "size_beta": positive(size_raw[..., 1]),
                "composition_concentration": positive(comp_raw),
            }

        def parse_guide_params(guide_params, image):
            per_slot, global_out = raw_guide_outputs(guide_params, image)
            return parse_raw(per_slot, global_out)

        def guide_distributions(parsed):
            return {
                "background_rgb": dist.TransformedDistribution(
                    dist.Beta(parsed["background_alpha"], parsed["background_beta"]),
                    dist.transforms.AffineTransform(bg_low, bg_high - bg_low),
                ).to_event(1),
                "bg_gradient": dist.Normal(parsed["grad_loc"], 0.05).to_event(2),
                "observation_noise": dist.LogNormal(parsed["noise_loc"], parsed["noise_scale"]),
                "presence": dist.Bernoulli(logits=parsed["presence_logits"]).to_event(1),
                "position": dist.TransformedDistribution(
                    dist.Beta(parsed["position_alpha"], parsed["position_beta"]),
                    dist.transforms.AffineTransform(position_low, position_scale),
                ).to_event(2),
                "size": dist.TransformedDistribution(
                    dist.Beta(parsed["size_alpha"], parsed["size_beta"]),
                    dist.transforms.AffineTransform(size_low, size_high - size_low),
                ).to_event(1),
                "composition": dist.Dirichlet(parsed["composition_concentration"]).to_event(1),
            }

        def init_guide_params(key):
            variables = guide_module.init(key, jnp.ones((1, *image_shape, channels), dtype=jnp.float32))
            return variables["params"]

        def guide(image, guide_params=None):
            image_batch, _ = ensure_image_batch(image)
            if guide_params is None:
                net = flax_module("v4_guide", guide_module, input_shape=(1, *image_shape, channels))
            else:
                with handlers.substitute(data={guide_param_name: guide_params}):
                    net = flax_module("v4_guide", guide_module, input_shape=(1, *image_shape, channels))
            per_slot, global_out = net(image_batch)
            distributions = guide_distributions(parse_raw(per_slot, global_out))
            with numpyro.plate("batch", image_batch.shape[0]):
                for name in sites:
                    numpyro.sample(name, distributions[name])

        def guide_log_prob(guide_params, image, latents):
            image_batch, was_single = ensure_image_batch(image)
            distributions = guide_distributions(parse_guide_params(guide_params, image_batch))
            terms = []
            for name in sites:
                value = jnp.asarray(latents[name])
                if was_single:
                    if name == "observation_noise" and value.ndim == 0:
                        value = value[None]
                    elif name == "background_rgb" and value.ndim == 1:
                        value = value[None, :]
                    elif name == "bg_gradient" and value.ndim == 2:
                        value = value[None, :, :]
                    elif name in ("presence", "size") and value.ndim == 1:
                        value = value[None, :]
                    elif name in ("position", "composition") and value.ndim == 2:
                        value = value[None, :, :]
                terms.append(distributions[name].log_prob(value))
            total = sum(terms)
            return total[0] if was_single else total

        def guide_point_estimates(guide_params, image):
            parsed = parse_guide_params(guide_params, image)
            bg_unit = parsed["background_alpha"] / (parsed["background_alpha"] + parsed["background_beta"])
            position_unit = parsed["position_alpha"] / (parsed["position_alpha"] + parsed["position_beta"])
            size_unit = parsed["size_alpha"] / (parsed["size_alpha"] + parsed["size_beta"])
            composition = parsed["composition_concentration"] / jnp.sum(
                parsed["composition_concentration"], axis=-1, keepdims=True
            )
            return {
                "background_rgb": bg_low + (bg_high - bg_low) * bg_unit,
                "bg_gradient": parsed["grad_loc"],
                "observation_noise": jnp.exp(parsed["noise_loc"] + 0.5 * parsed["noise_scale"] ** 2),
                "presence_probs": jnn.sigmoid(parsed["presence_logits"]),
                "position": position_low + position_scale * position_unit,
                "size": size_low + (size_high - size_low) * size_unit,
                "composition": composition,
            }

        def render_from_estimates(estimates, renderer_params=None, chunk=4):
            if renderer_params is None:
                renderer_params = initial_renderer_params
            n = estimates["background_rgb"].shape[0]
            outs = []
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                latents = {
                    "background_rgb": estimates["background_rgb"][start:end],
                    "bg_gradient": estimates["bg_gradient"][start:end],
                    "presence": estimates["presence_probs"][start:end],
                    "position": estimates["position"][start:end],
                    "size": estimates["size"][start:end],
                    "composition": estimates["composition"][start:end],
                }
                outs.append(render_latent_batch(latents, renderer_params))
            return jnp.concatenate(outs, axis=0)

        centre_slot = 27
        centre_py = int(float(position_center[centre_slot, 0]) * (image_shape[0] - 1))
        centre_px = int(float(position_center[centre_slot, 1]) * (image_shape[1] - 1))

        def render_single_object(composition_value, size_value=0.16, renderer_params=None):
            if renderer_params is None:
                renderer_params = initial_renderer_params
            presence = jnp.zeros(max_objects, dtype=jnp.float32).at[centre_slot].set(1.0)
            composition = (jnp.ones((max_objects, composition_dim), dtype=jnp.float32) / composition_dim).at[centre_slot].set(composition_value)
            return render_scene(
                jnp.array([0.6, 0.6, 0.55], dtype=jnp.float32),
                jnp.zeros((2, 3), dtype=jnp.float32),
                presence,
                position_center,
                size_value * jnp.ones(max_objects, dtype=jnp.float32),
                composition,
                renderer_params,
            )

        def composition_to_rgb(composition, renderer_params=None):
            # Fast diagnostic colour readout. Do not render a full 64x64 scene for
            # every composition in the validation set; query the object Flax module at
            # the object centre on a neutral local background.
            if renderer_params is None:
                renderer_params = initial_renderer_params
            comp = jnp.asarray(composition, dtype=jnp.float32)
            flat = comp.reshape(-1, composition_dim)
            local = jnp.zeros((flat.shape[0], 1, 1, 2), dtype=jnp.float32)
            size = 0.16 * jnp.ones((flat.shape[0],), dtype=jnp.float32)
            local_bg = 0.6 * jnp.ones((flat.shape[0], 1, 1, channels), dtype=jnp.float32)
            _, rgb = renderer_module.apply({"params": renderer_params}, local, size, flat, local_bg)
            return rgb[:, 0, 0, :].reshape(comp.shape[:-1] + (channels,))

        return ModelVersion(
            name="v4",
            description=(
                "Flax random-Fourier coordinate renderer, 8x8 cell-labelled free-placement prior, "
                "Flax slot-aligned NumPyro guide via flax_module"
            ),
            image_shape=image_shape,
            channels=channels,
            max_objects=max_objects,
            composition_dim=composition_dim,
            site_names=sites,
            position_low=position_low,
            position_high=position_high,
            size_low=size_low,
            size_high=size_high,
            model=model,
            init_guide_params=init_guide_params,
            guide=guide,
            guide_log_prob=guide_log_prob,
            guide_point_estimates=guide_point_estimates,
            render=render_from_estimates,
            model_log_joint=model_log_joint,
            composition_to_rgb=composition_to_rgb,
            predictive_sites=(*sites, "mean", "obs", "count"),
            extras={
                "object_chunk": object_chunk,
                "simulate_chunk": 64,
                "sc_chunk": 64,
                "position_center": position_center,
                "render_scene": render_scene,
                "render_single_object": render_single_object,
                "renderer_module": renderer_module,
                "renderer_params": initial_renderer_params,
                "initial_renderer_params": initial_renderer_params,
                "renderer_param_name": renderer_param_name,
                "guide_module": guide_module,
                "guide_param_name": guide_param_name,
                "batched_model": batched_model,
                "estimate_chunk": 32,
                "presence_prob": presence_prob,
            },
        )

    return (build_v4,)


@app.cell(hide_code=True)
def _(jnp, load_real_rgb_patches):
    # Separate, fixed-size pool of real patches for renderer pretraining (NOT tied
    # to the SC widget). Decoupled deliberately: pretraining wants a stable corpus
    # that does not change if a user adjusts run-config widgets.
    N_PRETRAIN_PATCHES = 256

    pretrain_patches_np, _pretrain_meta = load_real_rgb_patches(
        patch=64,
        n_patches=N_PRETRAIN_PATCHES,
        seed=1234,
    )
    pretrain_images = jnp.asarray(pretrain_patches_np)
    pretrain_images.shape

    return N_PRETRAIN_PATCHES, pretrain_images


@app.cell(hide_code=True)
def _(
    N_PRETRAIN_PATCHES,
    build_v4,
    dist,
    jax,
    jnp,
    np,
    numpyro,
    pretrain_images,
    random,
):
    # Renderer pretraining via AutoNormal SVI on real patches.
    #
    # A purpose-built `pretraining_model` whose only sample sites are the per-image
    # continuous latents on their NATURAL [0,1] (or stick-breaking) domain. AutoNormal
    # can then automatically discover the right bijector for each. Affine transforms
    # from unit-domain to physical-domain happen DETERMINISTICALLY outside the sample
    # sites. This avoids a bug where `TransformedDistribution(Beta, AffineTransform)`
    # exposes its support as `Real()` to `biject_to`, so AutoNormal samples on the
    # whole real line, sometimes producing negative sizes that crash the renderer.
    #
    # `presence` is replaced by a per-image, per-slot `numpyro.param` (soft, in [0,1])
    # so AutoNormal does not have to deal with a discrete site. `observation_noise`
    # is hard-coded so renderer cannot hide residuals by inflating sigma.
    #
    # After SVI we throw away the loc/scale params and the per-image presence
    # param; we keep the trained `v4_renderer$params` as the renderer init.

    _av_init = build_v4()
    _max_objects = _av_init.max_objects
    _composition_dim = _av_init.composition_dim
    _position_low = _av_init.position_low
    _position_high = _av_init.position_high
    _size_low = _av_init.size_low
    _size_high = _av_init.size_high
    _bg_low = jnp.array([0.40, 0.40, 0.35], dtype=jnp.float32)
    _bg_high = jnp.array([0.80, 0.80, 0.75], dtype=jnp.float32)
    _renderer_param_name = _av_init.extras["renderer_param_name"]
    _initial_renderer_params = _av_init.extras["initial_renderer_params"]
    _render_scene = _av_init.extras["render_scene"]
    _presence_prob = _av_init.extras["presence_prob"]

    def _render_batch(latents, renderer_params):
        return jax.vmap(lambda *args: _render_scene(*args, renderer_params))(
            latents["background_rgb"],
            latents["bg_gradient"],
            latents["presence"],
            latents["position"],
            latents["size"],
            latents["composition"],
        )

    PRETRAIN_STEPS = 1000
    PRETRAIN_SUBSAMPLE = 16
    PRETRAIN_LR = 1e-3
    PRETRAIN_SIGMA_OBS = 0.02

    def pretraining_model(full_images):
        renderer_params = numpyro.param(_renderer_param_name, _initial_renderer_params)
        soft_presence_param = numpyro.param(
            "pretrain_soft_presence",
            _presence_prob * jnp.ones((N_PRETRAIN_PATCHES, _max_objects), dtype=jnp.float32),
            constraint=dist.constraints.unit_interval,
        )
        with numpyro.plate("images", N_PRETRAIN_PATCHES, subsample_size=PRETRAIN_SUBSAMPLE) as idx:
            images_minibatch = full_images[idx]
            presence_minibatch = soft_presence_param[idx]
            # All latents on their natural [0,1] / stick-breaking domain so AutoNormal
            # gets correct bijectors.
            bg_unit = numpyro.sample(
                "bg_unit", dist.Beta(2.0, 2.0).expand([3]).to_event(1)
            )
            pos_unit = numpyro.sample(
                "pos_unit", dist.Beta(2.0, 2.0).expand([_max_objects, 2]).to_event(2)
            )
            size_unit = numpyro.sample(
                "size_unit", dist.Beta(1.3, 3.0).expand([_max_objects]).to_event(1)
            )
            bg_gradient = numpyro.sample(
                "bg_gradient", dist.Normal(0.0, 0.08).expand([2, 3]).to_event(2)
            )
            composition = numpyro.sample(
                "composition",
                dist.Dirichlet(jnp.ones(_composition_dim)).expand([_max_objects]).to_event(1),
            )
            # Deterministic affines to physical domain.
            background_rgb = _bg_low + (_bg_high - _bg_low) * bg_unit
            position = _position_low + (_position_high - _position_low) * pos_unit
            size = _size_low + (_size_high - _size_low) * size_unit
            latents = {
                "background_rgb": background_rgb,
                "bg_gradient": bg_gradient,
                "presence": presence_minibatch,
                "position": position,
                "size": size,
                "composition": composition,
            }
            mean = _render_batch(latents, renderer_params)
            numpyro.sample(
                "obs",
                dist.Normal(mean, PRETRAIN_SIGMA_OBS).to_event(3),
                obs=images_minibatch,
            )

    _autoguide = numpyro.infer.autoguide.AutoNormal(pretraining_model, init_scale=0.05)
    _pretrain_svi = numpyro.infer.SVI(
        pretraining_model,
        _autoguide,
        numpyro.optim.Adam(PRETRAIN_LR),
        numpyro.infer.Trace_ELBO(),
    )
    _pretrain_result = _pretrain_svi.run(
        random.PRNGKey(7777),
        PRETRAIN_STEPS,
        pretrain_images,
        progress_bar=False,
    )
    pretrain_losses = np.asarray(_pretrain_result.losses)
    pretrained_renderer_params = _pretrain_result.params[_renderer_param_name]

    _finite = pretrain_losses[np.isfinite(pretrain_losses)]
    pretrain_summary = {
        "N_patches": int(N_PRETRAIN_PATCHES),
        "subsample": int(PRETRAIN_SUBSAMPLE),
        "steps": int(PRETRAIN_STEPS),
        "lr": float(PRETRAIN_LR),
        "sigma_obs": float(PRETRAIN_SIGMA_OBS),
        "loss_first": float(pretrain_losses[0]),
        "loss_last": float(pretrain_losses[-1]),
        "loss_min": float(_finite.min()) if _finite.size else float("nan"),
        "n_nonfinite": int(np.sum(~np.isfinite(pretrain_losses))),
    }
    pretrain_summary

    return (pretrained_renderer_params,)


@app.cell(hide_code=True)
def _(build_v4, pretrained_renderer_params):
    active_version = build_v4(renderer_init=pretrained_renderer_params)

    return (active_version,)


@app.cell(hide_code=True)
def _(active_version, mo):
    mo.md(f"""
    **Active model:** `{active_version.name}` — {active_version.description}\n\n"
        f"Image `{active_version.image_shape}` RGB; max objects `{active_version.max_objects}`; "
        f"composition dim `{active_version.composition_dim}`; size support "
        f"`[{active_version.size_low}, {active_version.size_high}]`.
    """)
    return


@app.cell
def _(Predictive, block_until_ready_tree, gc, jnp, random, tqdm):
    def simulate_pairs(version, n, key, chunk=None, memory_log=None, label="simulate",
                       randomize_renderer_seed=False):
        """Generate (image, latent) training pairs by sampling from `version.model`.

        When `randomize_renderer_seed=True`, each chunk uses a freshly initialised
        renderer (a fresh draw of the renderer\'s Flax params from `renderer_module.init`
        with a chunk-specific PRNG key). This is "domain randomisation" for the
        appearance manifold: each batch of training images uses a different random
        renderer, so the union of training data spans a much wider colour /
        texture distribution than a single fixed renderer can. The guide is
        forced to learn renderer-invariant inferences (position, size, presence)
        rather than memorising one renderer\'s palette.

        Mechanism: `Predictive` substitutes its `.params` dict into the model on
        every call. We mutate `predictive.params` per chunk to override the
        renderer\'s `numpyro.param` site. No re-jit: `Predictive` keys its
        compilation on shapes, not values.
        """
        chunk = int(chunk or version.extras.get("simulate_chunk", 4))
        predictive = Predictive(
            version.model,
            num_samples=chunk,
            return_sites=version.predictive_sites,
        )
        # Pre-compute the neutral inputs used to init the renderer (need to match
        # the shapes `build_v4` uses for its own init).
        if randomize_renderer_seed:
            _rm = version.extras["renderer_module"]
            _object_chunk = version.extras["object_chunk"]
            _img_shape = version.image_shape
            _channels = version.channels
            _comp_dim = version.composition_dim
            _renderer_init_inputs = (
                jnp.zeros((_object_chunk, *_img_shape, 2), dtype=jnp.float32),
                0.12 * jnp.ones((_object_chunk,), dtype=jnp.float32),
                jnp.ones((_object_chunk, _comp_dim), dtype=jnp.float32) / _comp_dim,
                0.6 * jnp.ones((_object_chunk, *_img_shape, _channels), dtype=jnp.float32),
            )
        outs = []
        produced = 0
        num_chunks = int((n + chunk - 1) // chunk)
        for chunk_id in tqdm(range(num_chunks), desc=label, leave=False):
            if randomize_renderer_seed:
                # Fold in a chunk-specific seed for the renderer init; keep this stream
                # separate from the sampling stream so renderer randomness does not
                # alias with scene randomness.
                _renderer_key = random.fold_in(key, 100_000 + chunk_id)
                _fresh = _rm.init(_renderer_key, *_renderer_init_inputs)["params"]
                predictive.params = {version.extras["renderer_param_name"]: _fresh}
            draw = predictive(random.fold_in(key, chunk_id))
            block_until_ready_tree(draw["obs"])
            keep = min(chunk, n - produced)
            if keep < chunk:
                draw = {name: value[:keep] for name, value in draw.items()}
            outs.append(draw)
            produced += keep
            if memory_log is not None and (chunk_id == 0 or produced >= n):
                memory_log.mark(f"{label}:{produced}/{n}")
        result = {name: jnp.concatenate([draw[name] for draw in outs], axis=0) for name in outs[0]}
        block_until_ready_tree(result["obs"])
        del outs
        gc.collect()
        return result


    return (simulate_pairs,)


@app.cell
def _(jax, jnp, np, optax, random, tqdm):
    def npe_loss(version, guide_params, batch):
        latents = {name: batch[name] for name in version.site_names}
        return -jnp.mean(version.guide_log_prob(guide_params, batch["obs"], latents))

    def make_npe_trainer(version, lr=5e-4):
        optimizer = optax.adam(lr)

        @jax.jit
        def step(guide_params, opt_state, batch):
            loss_value, gradients = jax.value_and_grad(lambda params: npe_loss(version, params, batch))(guide_params)
            updates, opt_state = optimizer.update(gradients, opt_state, guide_params)
            guide_params = optax.apply_updates(guide_params, updates)
            return guide_params, opt_state, loss_value

        return optimizer, step

    def eval_loss_chunked(version, guide_params, data, chunk=64):
        n = data["obs"].shape[0]
        total = 0.0
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            batch = {name: value[start:end] for name, value in data.items()}
            total += float(npe_loss(version, guide_params, batch)) * (end - start)
        return total / n

    def train_npe(
        version,
        train_data,
        val_data,
        steps,
        batch_size,
        lr,
        key,
        eval_every,
        eval_chunk,
        memory_log=None,
    ):
        optimizer, step = make_npe_trainer(version, lr)
        guide_params = version.init_guide_params(random.fold_in(key, 1))
        opt_state = optimizer.init(guide_params)
        n_train = train_data["obs"].shape[0]
        init_val = eval_loss_chunked(version, guide_params, val_data, eval_chunk)
        best = (init_val, guide_params, 0)
        history = [(0, np.nan, init_val)]
        if memory_log is not None:
            memory_log.mark("after_guide_init")
        for step_id in tqdm(range(1, steps + 1), desc="NPE", miniters=max(1, steps // 10), leave=False):
            draw_key = random.fold_in(key, 1000 + step_id)
            replace = batch_size > n_train
            indices = random.choice(draw_key, n_train, (batch_size,), replace=replace)
            batch = {name: value[indices] for name, value in train_data.items()}
            guide_params, opt_state, train_loss = step(guide_params, opt_state, batch)
            if step_id % eval_every == 0 or step_id == steps:
                val_loss = eval_loss_chunked(version, guide_params, val_data, eval_chunk)
                train_loss_float = float(train_loss)
                history.append((step_id, train_loss_float, val_loss))
                if val_loss < best[0]:
                    best = (val_loss, guide_params, step_id)
                if memory_log is not None:
                    memory_log.mark(f"train_step:{step_id}")
        return best[1], history, {"init_val": init_val, "best_val": best[0], "best_step": best[2]}

    return (train_npe,)


@app.cell
def _(np, plt):
    def show_img(ax, image):
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 2:
            ax.imshow(np.clip(arr, 0.0, 1.0), cmap="magma", interpolation="nearest")
        else:
            ax.imshow(np.clip(arr, 0.0, 1.0), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    def circle_radius(version, size):
        return float(size) * max(version.image_shape)

    def add_true_circles(ax, version, presence, position, size, colour="cyan", linewidth=0.9):
        for obj in range(version.max_objects):
            if float(presence[obj]) > 0.5:
                pos = np.asarray(position[obj])
                ax.add_patch(
                    plt.Circle(
                        (pos[1] * (version.image_shape[1] - 1), pos[0] * (version.image_shape[0] - 1)),
                        radius=circle_radius(version, size[obj]),
                        edgecolor=colour,
                        facecolor="none",
                        linewidth=linewidth,
                    )
                )

    def add_pred_circles(ax, version, probs, position, size, threshold=0.35, linewidth=0.75):
        for obj in range(version.max_objects):
            prob = float(probs[obj])
            if prob > threshold:
                pos = np.asarray(position[obj])
                ax.add_patch(
                    plt.Circle(
                        (pos[1] * (version.image_shape[1] - 1), pos[0] * (version.image_shape[0] - 1)),
                        radius=circle_radius(version, size[obj]),
                        edgecolor=(1.0, 0.0, 1.0, max(0.15, min(1.0, prob))),
                        facecolor="none",
                        linewidth=linewidth,
                    )
                )

    return add_pred_circles, add_true_circles, show_img


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior predictive diagnostics and visualisations
    """)
    return


@app.cell
def _(MemoryLog, active_version, random, run_config, simulate_pairs):
    memory_log = MemoryLog()
    memory_log.mark("start")
    prior_visual_data = simulate_pairs(
        active_version,
        run_config.prior_visual_samples,
        random.PRNGKey(10),
        chunk=min(run_config.simulate_chunk, active_version.extras["simulate_chunk"]),
        memory_log=memory_log,
        label="prior",
    )
    memory_log.mark("after_prior_visual")
    return memory_log, prior_visual_data


@app.cell
def _(active_version, np, plt, prior_visual_data, show_img):
    plot_n = min(16, prior_visual_data["obs"].shape[0])
    _rows_prior = int(np.ceil(plot_n / 4))
    fig_prior, axes_prior = plt.subplots(_rows_prior, 4, figsize=(9, 2.25 * _rows_prior))
    _axes_flat_prior = np.asarray(axes_prior).reshape(-1)
    for _ax, _image, _count in zip(_axes_flat_prior, np.asarray(prior_visual_data["obs"][:plot_n]), np.asarray(prior_visual_data["count"][:plot_n])):
        show_img(_ax, _image)
        _ax.set_title(f"count={int(_count)}", fontsize=8)
    for _ax in _axes_flat_prior[plot_n:]:
        _ax.axis("off")
    fig_prior.suptitle(f"{active_version.name} prior predictive RGB patches", y=0.995)
    fig_prior.tight_layout()
    fig_prior
    return


@app.cell
def _(np, plt, prior_visual_data, show_img):
    compare_n = min(6, prior_visual_data["obs"].shape[0])
    fig_prior_compare, axes_prior_compare = plt.subplots(compare_n, 3, figsize=(8, 2.25 * compare_n))
    axes_prior_compare = np.asarray(axes_prior_compare).reshape(compare_n, 3)
    for _row in range(compare_n):
        _mean_img = np.asarray(prior_visual_data["mean"][_row])
        _obs_img = np.asarray(prior_visual_data["obs"][_row])
        _residual = np.abs(_obs_img - _mean_img)
        for _col, (_image, _title) in enumerate([(_mean_img, "mean"), (_obs_img, "observed"), (_residual, "|obs-mean|")]):
            show_img(axes_prior_compare[_row, _col], _image)
            axes_prior_compare[_row, _col].set_title(_title if _row == 0 else "", fontsize=9)
    fig_prior_compare.suptitle("Prior predictive: deterministic mean vs noisy observation", y=0.998)
    fig_prior_compare.tight_layout()
    fig_prior_compare
    return


@app.cell
def _(active_version, jnp, np, plt, show_img):
    sprite_compositions = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.65, 0.25, 0.10],
        ],
        dtype=jnp.float32,
    )
    sprite_sizes = jnp.array([0.06, 0.14, 0.26], dtype=jnp.float32)
    fig_sprites, axes_sprites = plt.subplots(len(sprite_compositions), len(sprite_sizes), figsize=(7.5, 8.5))
    for _row in range(len(sprite_compositions)):
        for _col in range(len(sprite_sizes)):
            _image = active_version.extras["render_single_object"](sprite_compositions[_row], float(sprite_sizes[_col]))
            show_img(axes_sprites[_row, _col], _image)
            axes_sprites[_row, _col].set_title(
                f"comp={np.asarray(sprite_compositions[_row]).round(2)}\nsize={float(sprite_sizes[_col]):.2f}",
                fontsize=8,
            )
    fig_sprites.suptitle("Single-object Flax renderer: composition and size effects", y=0.995)
    fig_sprites.tight_layout()
    fig_sprites
    return


@app.cell
def _(active_version, jnp, np, plt, prior_visual_data):
    presence_np = np.asarray(prior_visual_data["presence"])
    active_mask = presence_np.astype(bool)
    active_positions = np.asarray(prior_visual_data["position"])[active_mask]
    active_sizes = np.asarray(prior_visual_data["size"])[active_mask]
    active_compositions = np.asarray(prior_visual_data["composition"])[active_mask]
    active_colours = np.asarray(active_version.composition_to_rgb(jnp.asarray(active_compositions)))
    counts_np = np.asarray(prior_visual_data["count"])

    fig_latents, axes_latents = plt.subplots(2, 3, figsize=(11, 6))
    axes_latents[0, 0].hist(counts_np, bins=np.arange(-0.5, active_version.max_objects + 1.5), rwidth=0.8)
    axes_latents[0, 0].set_title("object count")
    axes_latents[0, 0].set_xlim(-0.5, max(15, np.max(counts_np) + 1.5))
    axes_latents[0, 1].scatter(active_positions[:, 1], active_positions[:, 0], s=8, alpha=0.6)
    axes_latents[0, 1].invert_yaxis()
    axes_latents[0, 1].set_title("active positions")
    axes_latents[0, 1].set_xlabel("x")
    axes_latents[0, 1].set_ylabel("y")
    axes_latents[0, 2].hist(active_sizes, bins=20, color="tab:green")
    axes_latents[0, 2].set_title("active sizes")
    for k in range(active_version.composition_dim):
        axes_latents[1, 0].hist(active_compositions[:, k], bins=20, alpha=0.55, label=f"c{k}")
    axes_latents[1, 0].set_title("composition simplex components")
    axes_latents[1, 0].legend(fontsize=8)
    axes_latents[1, 1].scatter(active_colours[:, 0], active_colours[:, 2], c=np.clip(active_colours, 0, 1), s=16)
    axes_latents[1, 1].set_title("composition-implied colours")
    axes_latents[1, 1].set_xlabel("red")
    axes_latents[1, 1].set_ylabel("blue")
    axes_latents[1, 2].hist(np.asarray(prior_visual_data["observation_noise"]), bins=20, color="tab:orange")
    axes_latents[1, 2].set_title("observation noise")
    fig_latents.suptitle("Prior latent distributions", y=0.995)
    fig_latents.tight_layout()
    fig_latents
    return


@app.cell
def _(active_version, add_true_circles, np, plt, prior_visual_data, show_img):
    overlay_n = min(8, prior_visual_data["obs"].shape[0])
    fig_overlay, axes_overlay = plt.subplots(2, 4, figsize=(10, 5.2))
    _axes_flat_overlay = axes_overlay.ravel()
    for _ax, _idx in zip(_axes_flat_overlay, range(overlay_n)):
        show_img(_ax, prior_visual_data["obs"][_idx])
        _ax.set_title(f"sample {_idx}; count={int(prior_visual_data['count'][_idx])}", fontsize=8)
        add_true_circles(
            _ax,
            active_version,
            np.asarray(prior_visual_data["presence"][_idx]),
            np.asarray(prior_visual_data["position"][_idx]),
            np.asarray(prior_visual_data["size"][_idx]),
            colour="cyan",
        )
    for _ax in _axes_flat_overlay[overlay_n:]:
        _ax.axis("off")
    fig_overlay.suptitle("Prior images with true position/size overlays", y=1.02)
    fig_overlay.tight_layout()
    fig_overlay
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chunked simulation and NPE training

    The training objective is still exact NPE, `-log q_phi(theta_sim | x_sim)`, not
    reconstruction. Validation is evaluated in chunks. Script smoke runs are intentionally
    tiny; choose the `useful` profile widget for a more meaningful run.
    """)
    return


@app.cell
def _(
    active_version,
    memory_log,
    random,
    run_config,
    simulate_pairs,
    train_npe,
):
    if run_config.run_training:
        train_key = random.PRNGKey(100)
        train_data = simulate_pairs(
            active_version,
            run_config.train_size,
            random.fold_in(train_key, 1),
            chunk=run_config.simulate_chunk,
            memory_log=memory_log,
            label="train-sim",
        )
        memory_log.mark("after_train_sim")
        val_data = simulate_pairs(
            active_version,
            run_config.val_size,
            random.fold_in(train_key, 2),
            chunk=run_config.simulate_chunk,
            memory_log=memory_log,
            label="val-sim",
        )
        memory_log.mark("after_val_sim")
        trained_guide_params, training_history, training_summary = train_npe(
            active_version,
            train_data,
            val_data,
            steps=run_config.train_steps,
            batch_size=run_config.batch_size,
            lr=run_config.learning_rate,
            key=train_key,
            eval_every=run_config.eval_every,
            eval_chunk=run_config.eval_chunk,
            memory_log=memory_log,
        )
        memory_log.mark("after_training")
    else:
        train_data = val_data = trained_guide_params = None
        training_history = []
        training_summary = {"training": "disabled"}
    training_summary
    return train_data, trained_guide_params, training_history, val_data


@app.cell(hide_code=True)
def _(mo, train_data):
    # Scrollable inspector for the training set.
    #
    # Shows, side-by-side: the training image, the guide\'s per-image render
    # (using both the original-init renderer AND the SVI-trained renderer for
    # comparison), the residual, true vs. inferred slot overlays, and the
    # inferred colour palette as RGB swatches so the "always yellow" symptom
    # is directly visible.
    train_idx_slider = mo.ui.slider(
        start=0,
        stop=int(train_data["obs"].shape[0]) - 1,
        step=1,
        value=0,
        label="train index",
        show_value=True,
        full_width=True,
    )
    train_idx_slider

    return (train_idx_slider,)


@app.cell(hide_code=True)
def _(
    active_version,
    add_pred_circles,
    add_true_circles,
    jnp,
    np,
    plt,
    show_img,
    train_data,
    train_idx_slider,
    trained_guide_params,
    trained_renderer_params,
):
    _idx = int(train_idx_slider.value)
    _img = train_data["obs"][_idx]
    _true_latents = {name: train_data[name][_idx] for name in active_version.site_names}
    _true_mean = train_data["mean"][_idx]

    # Guide point estimates on this single image.
    _img_batch = _img[None]
    _est = active_version.guide_point_estimates(trained_guide_params, _img_batch)

    # Two renders: with the SVI-trained renderer and with the deterministic init.
    _fit_svi = active_version.render(_est, renderer_params=trained_renderer_params)[0]
    _fit_init = active_version.render(_est)[0]
    _residual_svi = np.abs(np.asarray(_img) - np.asarray(_fit_svi))

    # Inferred composition colours per slot, masked by inferred presence (using SVI renderer).
    _inferred_comp = np.asarray(_est["composition"][0])              # (max_objects, comp_dim)
    _inferred_presence = np.asarray(_est["presence_probs"][0])       # (max_objects,)
    _inferred_colours_svi = np.asarray(active_version.composition_to_rgb(
        jnp.asarray(_inferred_comp), renderer_params=trained_renderer_params
    ))  # (max_objects, 3)
    _inferred_colours_init = np.asarray(active_version.composition_to_rgb(jnp.asarray(_inferred_comp)))
    # True colours from this image\'s true composition.
    _true_comp = np.asarray(_true_latents["composition"])
    _true_presence = np.asarray(_true_latents["presence"])
    _true_colours = np.asarray(active_version.composition_to_rgb(jnp.asarray(_true_comp)))

    # Layout: top row = images, bottom row = colour swatches per active slot.
    fig_train_viewer, axes_tv = plt.subplots(2, 4, figsize=(11, 5.5), dpi=90,
                                              gridspec_kw={"height_ratios": [3, 1]})
    # Row 0: image / fit-init / fit-svi / residual.
    show_img(axes_tv[0, 0], _img)
    axes_tv[0, 0].set_title(f"train[{_idx}] (true mean overlaid in cyan)", fontsize=9)
    show_img(axes_tv[0, 1], _fit_init)
    axes_tv[0, 1].set_title("guide z -> init renderer", fontsize=9)
    show_img(axes_tv[0, 2], _fit_svi)
    axes_tv[0, 2].set_title("guide z -> SVI renderer", fontsize=9)
    show_img(axes_tv[0, 3], _residual_svi)
    axes_tv[0, 3].set_title("|residual| (SVI render)", fontsize=9)

    # True slots cyan, predicted slots magenta-by-prob.
    add_true_circles(axes_tv[0, 0], active_version, _true_presence,
                     _true_latents["position"], _true_latents["size"])
    add_pred_circles(axes_tv[0, 2], active_version, _inferred_presence,
                     _est["position"][0], _est["size"][0])

    # Row 1: colour swatches.
    # Cell 1-0: true colours of active true slots.
    # Cell 1-1: inferred colours via init renderer (only for active inferred slots).
    # Cell 1-2: inferred colours via SVI renderer (only for active inferred slots).
    # Cell 1-3: composition simplex scatter (true vs inferred for active inferred slots).
    def _swatch_row(ax, colours, mask=None, title=""):
        """Draw colours as a horizontal swatch strip. mask filters which to show."""
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        if mask is not None:
            idx = np.where(mask > 0.5)[0]
            colours = colours[idx]
        if len(colours) == 0:
            ax.text(0.5, 0.5, "no active slots", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            return
        n = len(colours)
        for i, c in enumerate(np.clip(colours, 0, 1)):
            ax.add_patch(plt.Rectangle((i / n, 0), 1.0 / n, 1.0, color=c))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    _active_true_mask = _true_presence.astype(bool)
    _active_pred_mask = (_inferred_presence > 0.35)
    _swatch_row(axes_tv[1, 0], _true_colours, mask=_active_true_mask,
                title=f"true colours ({int(_active_true_mask.sum())} slots)")
    _swatch_row(axes_tv[1, 1], _inferred_colours_init, mask=_active_pred_mask,
                title=f"inferred via init renderer ({int(_active_pred_mask.sum())} slots)")
    _swatch_row(axes_tv[1, 2], _inferred_colours_svi, mask=_active_pred_mask,
                title=f"inferred via SVI renderer ({int(_active_pred_mask.sum())} slots)")

    # Composition simplex plot: project (c0, c1) ignoring c2 = 1 - c0 - c1.
    ax_simplex = axes_tv[1, 3]
    ax_simplex.set_xticks([]); ax_simplex.set_yticks([])
    ax_simplex.set_title("composition (c0 vs c1)", fontsize=8)
    # Reference simplex border.
    ax_simplex.plot([0, 1, 0, 0], [0, 0, 1, 0], color="grey", linewidth=0.7)
    if _active_true_mask.sum() > 0:
        ax_simplex.scatter(_true_comp[_active_true_mask, 0], _true_comp[_active_true_mask, 1],
                           c=np.clip(_true_colours[_active_true_mask], 0, 1),
                           edgecolors="cyan", linewidths=1.0, s=50, label="true")
    if _active_pred_mask.sum() > 0:
        ax_simplex.scatter(_inferred_comp[_active_pred_mask, 0], _inferred_comp[_active_pred_mask, 1],
                           c=np.clip(_inferred_colours_svi[_active_pred_mask], 0, 1),
                           edgecolors="magenta", linewidths=1.0, s=50, marker="s", label="pred")
    ax_simplex.set_xlim(-0.05, 1.05); ax_simplex.set_ylim(-0.05, 1.05)
    ax_simplex.legend(fontsize=6, loc="upper right")

    fig_train_viewer.suptitle(
        f"Training image inspector — slot circles: cyan=true (left), magenta=inferred (right)",
        y=0.998, fontsize=10,
    )
    fig_train_viewer.tight_layout()
    fig_train_viewer

    return


@app.cell
def _(np, plt, training_history):
    if training_history:
        history_array = np.asarray(training_history, dtype=float)
        fig_loss, ax_loss = plt.subplots(figsize=(6.5, 3.5))
        if np.any(np.isfinite(history_array[:, 1])):
            ax_loss.plot(history_array[:, 0], history_array[:, 1], marker="o", label="mini-batch train")
        ax_loss.plot(history_array[:, 0], history_array[:, 2], marker="o", label="held-out validation")
        ax_loss.set_xlabel("training step")
        ax_loss.set_ylabel("NPE loss = -log q")
        ax_loss.set_title("Flax/NumPyro NPE training")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend()
        fig_loss.tight_layout()
        training_loss_plot = fig_loss
    else:
        training_loss_plot = None
    training_loss_plot
    return


@app.cell
def _(jnp, np):
    def latent_diagnostics(version, guide_params, data, n_eval=None):
        n = data["obs"].shape[0] if n_eval is None else min(n_eval, data["obs"].shape[0])
        eval_data = {name: value[:n] for name, value in data.items()}
        estimate_chunk = int(version.extras.get("estimate_chunk", 32))
        estimate_parts = []
        for start in range(0, n, estimate_chunk):
            end = min(start + estimate_chunk, n)
            estimate_parts.append(version.guide_point_estimates(guide_params, eval_data["obs"][start:end]))
        estimates = {
            name: jnp.concatenate([part[name] for part in estimate_parts], axis=0)
            for name in estimate_parts[0]
        }
        presence = eval_data["presence"]
        active_count = jnp.sum(presence)
        position_error_slot = jnp.sqrt(jnp.sum((estimates["position"] - eval_data["position"]) ** 2, axis=-1))
        size_error_slot = jnp.abs(estimates["size"] - eval_data["size"])
        composition_error_slot = jnp.mean(jnp.abs(estimates["composition"] - eval_data["composition"]), axis=-1)
        position_mae = float(jnp.sum(position_error_slot * presence) / (active_count + 1e-6))
        size_mae = float(jnp.sum(size_error_slot * presence) / (active_count + 1e-6))
        composition_mae = float(jnp.sum(composition_error_slot * presence) / (active_count + 1e-6))
        presence_mae = float(jnp.mean(jnp.abs(estimates["presence_probs"] - presence)))
        expected_count = jnp.sum(estimates["presence_probs"], axis=1)
        count_mae_per = jnp.abs(expected_count - eval_data["count"])
        hard_count = jnp.sum((estimates["presence_probs"] > 0.5).astype(jnp.float32), axis=1)
        true_colours = version.composition_to_rgb(eval_data["composition"])
        pred_colours = version.composition_to_rgb(estimates["composition"])
        active_mask = presence.astype(bool)
        colour_mae = float(jnp.mean(jnp.abs(true_colours[active_mask] - pred_colours[active_mask])))
        colour_std_ratio = float(
            jnp.mean(jnp.std(pred_colours[active_mask], axis=0) / (jnp.std(true_colours[active_mask], axis=0) + 1e-6))
        )

        centre = jnp.asarray(0.5 * (np.asarray(version.position_low) + np.asarray(version.position_high)))
        base_position = float(jnp.sum(jnp.sqrt(jnp.sum((centre - eval_data["position"]) ** 2, axis=-1)) * presence) / (active_count + 1e-6))
        base_size = float(jnp.sum(jnp.abs(0.5 * (version.size_low + version.size_high) - eval_data["size"]) * presence) / (active_count + 1e-6))
        base_comp = float(jnp.sum(jnp.mean(jnp.abs(1.0 / version.composition_dim - eval_data["composition"]), axis=-1) * presence) / (active_count + 1e-6))
        base_count = float(jnp.mean(jnp.abs(version.extras["presence_prob"] * version.max_objects - eval_data["count"])))

        safe_count = jnp.maximum(eval_data["count"], 1.0)
        per_image_position = jnp.sum(position_error_slot * presence, axis=1) / safe_count
        per_image_size = jnp.sum(size_error_slot * presence, axis=1) / safe_count
        per_image_comp = jnp.sum(composition_error_slot * presence, axis=1) / safe_count
        per_image_score = per_image_position / (base_position + 1e-6) + per_image_size / (base_size + 1e-6) + per_image_comp / (base_comp + 1e-6) + count_mae_per / (base_count + 1e-6)

        count_strata = {}
        max_count_to_report = int(min(12, np.max(np.asarray(eval_data["count"])) if n else 0))
        for count_value in range(1, max_count_to_report + 1):
            mask = eval_data["count"] == count_value
            if bool(jnp.any(mask)):
                count_strata[count_value] = {
                    "n": int(jnp.sum(mask)),
                    "pos": float(jnp.mean(per_image_position[mask])),
                    "size": float(jnp.mean(per_image_size[mask])),
                    "comp": float(jnp.mean(per_image_comp[mask])),
                    "count_mae": float(jnp.mean(count_mae_per[mask])),
                    "hard_count_acc": float(jnp.mean(hard_count[mask] == eval_data["count"][mask])),
                }

        metrics = {
            "position_mae": position_mae,
            "size_mae": size_mae,
            "composition_mae": composition_mae,
            "presence_mae": presence_mae,
            "count_mae": float(jnp.mean(count_mae_per)),
            "hard_count_accuracy": float(jnp.mean(hard_count == eval_data["count"])),
            "colour_mae": colour_mae,
            "colour_std_ratio": colour_std_ratio,
            "base_position": base_position,
            "base_size": base_size,
            "base_comp": base_comp,
            "base_count": base_count,
            "mean_expected_count": float(jnp.mean(expected_count)),
            "mean_true_count": float(jnp.mean(eval_data["count"])),
            "count_strata": count_strata,
        }
        per_image = {
            "position": per_image_position,
            "size": per_image_size,
            "composition": per_image_comp,
            "count_mae": count_mae_per,
            "score": per_image_score,
        }
        return {"metrics": metrics, "estimates": estimates, "data": eval_data, "per_image": per_image}

    return (latent_diagnostics,)


@app.cell
def _(
    active_version,
    latent_diagnostics,
    train_data,
    trained_guide_params,
    val_data,
):
    if trained_guide_params is not None:
        train_diagnostics = latent_diagnostics(active_version, trained_guide_params, train_data)
        val_diagnostics = latent_diagnostics(active_version, trained_guide_params, val_data)
        diagnostics_summary = {
            "train": train_diagnostics["metrics"],
            "val": val_diagnostics["metrics"],
        }
    else:
        train_diagnostics = val_diagnostics = None
        diagnostics_summary = {"diagnostics": "training disabled"}
    diagnostics_summary
    return train_diagnostics, val_diagnostics


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Ground-truth latent recovery (synthetic)

    "
        "Gate on these latent metrics, not image MSE. Position/size/composition should beat the prior baselines; "
        "presence/count is reported explicitly because it is the known weak point of the free-placement v4 design.

    "
        f"`{diagnostics_summary}`
    """)
    return


@app.cell
def _(add_pred_circles, add_true_circles, np, plt, show_img):
    def plot_fit_examples(version, diagnostics, title, indices=None):
        data = diagnostics["data"]
        estimates = diagnostics["estimates"]
        if indices is None:
            indices = np.arange(min(6, data["obs"].shape[0]))
        indices = np.asarray(indices, dtype=int)
        subset_estimates = {name: value[indices] for name, value in estimates.items()}
        fit = version.render(subset_estimates)
        rows = len(indices)
        fig, axes = plt.subplots(rows, 3, figsize=(8.2, 2.35 * rows))
        axes = np.asarray(axes).reshape(rows, 3)
        for row, idx in enumerate(indices):
            obs = np.asarray(data["obs"][idx]).clip(0, 1)
            mean = np.asarray(data["mean"][idx]).clip(0, 1)
            pred = np.asarray(fit[row]).clip(0, 1)
            for col, (image, col_title) in enumerate([(obs, "obs + true"), (mean, "true mean"), (pred, "guide render + inferred")]):
                show_img(axes[row, col], image)
                axes[row, col].set_title(col_title if row == 0 else "", fontsize=9)
            add_true_circles(
                axes[row, 0], version,
                np.asarray(data["presence"][idx]),
                np.asarray(data["position"][idx]),
                np.asarray(data["size"][idx]),
            )
            add_pred_circles(
                axes[row, 2], version,
                np.asarray(estimates["presence_probs"][idx]),
                np.asarray(estimates["position"][idx]),
                np.asarray(estimates["size"][idx]),
            )
            axes[row, 0].set_ylabel(f"idx={idx}\ncount={int(data['count'][idx])}", fontsize=8)
        fig.suptitle(title, y=0.998)
        fig.tight_layout()
        return fig

    return (plot_fit_examples,)


@app.cell
def _(active_version, plot_fit_examples, train_diagnostics):
    if train_diagnostics is not None:
        fig_train_examples = plot_fit_examples(
            active_version,
            train_diagnostics,
            "Training-set sanity examples: true objects cyan, inferred slots magenta",
        )
    else:
        fig_train_examples = None
    fig_train_examples
    return


@app.cell
def _(active_version, plot_fit_examples, val_diagnostics):
    if val_diagnostics is not None:
        fig_val_examples = plot_fit_examples(
            active_version,
            val_diagnostics,
            "Held-out synthetic examples: true objects cyan, inferred slots magenta",
        )
    else:
        fig_val_examples = None
    fig_val_examples
    return


@app.cell
def _(active_version, np, plot_fit_examples, val_diagnostics):
    if val_diagnostics is not None:
        counts = np.asarray(val_diagnostics["data"]["count"])
        scores = np.asarray(val_diagnostics["per_image"]["score"])
        nonempty = np.where(counts > 0)[0]
        if len(nonempty) > 0:
            order = nonempty[np.argsort(scores[nonempty])]
            best_nonempty_indices = order[: min(6, len(order))]
            worst_nonempty_indices = order[-min(6, len(order)):][::-1]
            fig_best_latent = plot_fit_examples(
                active_version,
                val_diagnostics,
                "Best non-empty held-out examples ranked by ground-truth latent error",
                best_nonempty_indices,
            )
            fig_worst_latent = plot_fit_examples(
                active_version,
                val_diagnostics,
                "Worst non-empty held-out examples ranked by ground-truth latent error",
                worst_nonempty_indices,
            )
        else:
            best_nonempty_indices = worst_nonempty_indices = np.array([], dtype=int)
            fig_best_latent = fig_worst_latent = None
    else:
        best_nonempty_indices = worst_nonempty_indices = np.array([], dtype=int)
        fig_best_latent = fig_worst_latent = None
    fig_best_latent
    return (fig_worst_latent,)


@app.cell
def _(fig_worst_latent):
    fig_worst_latent
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real-image crops and optional self-consistency

    Native 64×64 RGB crops from `example.jpg`, no downsampling and no colour normalisation.
    The pre-SC panel is always shown after training. SC adaptation is still optional and
    conservative (enable the **Run real-image SC** widget); its log-joint rendering is chunked one flattened
    proposal/image pair at a time.
    """)
    return


@app.cell
def _(np):
    def load_real_rgb_patches(path="example.jpg", patch=64, n_patches=6, seed=0):
        from PIL import Image

        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        rng = np.random.default_rng(seed)
        patches, metadata = [], []
        for _ in range(n_patches):
            y0 = int(rng.integers(0, height - patch))
            x0 = int(rng.integers(0, width - patch))
            patches.append(image[y0:y0 + patch, x0:x0 + patch].astype(np.float32))
            metadata.append({"y": y0, "x": x0})
        return np.stack(patches), metadata

    return (load_real_rgb_patches,)


@app.cell
def _(
    active_version,
    jnp,
    load_real_rgb_patches,
    np,
    plt,
    run_config,
    show_img,
):
    real_patches_np, real_patch_meta = load_real_rgb_patches(
        patch=active_version.image_shape[0],
        n_patches=max(1, run_config.sc_patches),
        seed=4,
    )
    real_images = jnp.asarray(real_patches_np)
    real_summary = {
        "shape": tuple(real_images.shape),
        "min": float(jnp.min(real_images)),
        "max": float(jnp.max(real_images)),
        "channel_mean": np.asarray(jnp.mean(real_images, axis=(0, 1, 2))).round(3).tolist(),
        "channel_std": np.asarray(jnp.std(real_images, axis=(0, 1, 2))).round(3).tolist(),
        "metadata": real_patch_meta,
    }
    _cols_real = min(4, real_images.shape[0])
    _rows_real = int(np.ceil(real_images.shape[0] / _cols_real))
    fig_real, axes_real = plt.subplots(_rows_real, _cols_real, figsize=(2.4 * _cols_real, 2.4 * _rows_real))
    _axes_flat_real = np.asarray(axes_real).reshape(-1)
    for _ax, _idx in zip(_axes_flat_real, range(real_images.shape[0])):
        show_img(_ax, real_images[_idx])
        _ax.set_title(f"real {_idx}", fontsize=8)
    for _ax in _axes_flat_real[real_images.shape[0]:]:
        _ax.axis("off")
    fig_real.suptitle("Real native 64×64 RGB crops (no resize, no preprocessing)", y=0.995)
    fig_real.tight_layout()
    fig_real
    return (real_images,)


@app.cell
def _(
    active_version,
    add_pred_circles,
    np,
    plt,
    real_images,
    show_img,
    trained_guide_params,
):
    if trained_guide_params is not None:
        real_estimates_pre_sc = active_version.guide_point_estimates(trained_guide_params, real_images)
        real_fit_pre_sc = active_version.render(real_estimates_pre_sc)
        real_residual_pre_sc = np.abs(np.asarray(real_images) - np.asarray(real_fit_pre_sc))
        _rows_real_fit = real_images.shape[0]
        fig_real_fit, axes_real_fit = plt.subplots(_rows_real_fit, 3, figsize=(8, 2.35 * _rows_real_fit))
        axes_real_fit = np.asarray(axes_real_fit).reshape(_rows_real_fit, 3)
        for _row in range(_rows_real_fit):
            for _col, (_image, _title) in enumerate(
                [(real_images[_row], "real + inferred slots"), (real_fit_pre_sc[_row], "pre-SC guide render"), (real_residual_pre_sc[_row], "|residual|")]
            ):
                show_img(axes_real_fit[_row, _col], _image)
                axes_real_fit[_row, _col].set_title(_title if _row == 0 else "", fontsize=9)
            add_pred_circles(
                axes_real_fit[_row, 0],
                active_version,
                np.asarray(real_estimates_pre_sc["presence_probs"][_row]),
                np.asarray(real_estimates_pre_sc["position"][_row]),
                np.asarray(real_estimates_pre_sc["size"][_row]),
            )
        fig_real_fit.suptitle("Trained synthetic guide applied to real patches before SC", y=0.998)
        fig_real_fit.tight_layout()
    else:
        real_estimates_pre_sc = real_fit_pre_sc = None
        fig_real_fit = None
    fig_real_fit
    return real_estimates_pre_sc, real_fit_pre_sc


@app.cell(hide_code=True)
def _(
    active_version,
    handlers,
    jnp,
    np,
    random,
    real_images,
    trained_guide_params,
):
    # Step 2 -- SVI on real patches to learn the renderer.
    #
    # Only the renderer\'s Flax params (`v4_renderer$params`) are updated. The guide
    # stays frozen at the NPE-trained values; `observation_noise` is conditioned to
    # a fixed value (0.02) so the renderer can\'t trivially explain residuals by
    # inflating sigma; `presence` is conditioned to the guide\'s soft point estimate
    # so SVI sees no discrete latent variables.
    #
    # Loss: TraceMeanField_ELBO (closed-form KL where conjugate, sample-based
    # elsewhere). Optimizer: numpyro.optim.Adam(1e-4). Steps: 100. Runs end-to-end
    # in ~15s on this server.
    if trained_guide_params is not None:
        _svi_lr = 1e-4
        _svi_steps = 250
        _svi_sigma = 0.02
        _svi_seed = 123

        _bm = active_version.extras["batched_model"]
        _B = real_images.shape[0]
        _fixed_noise = _svi_sigma * jnp.ones(_B)
        _soft_presence = active_version.guide_point_estimates(
            trained_guide_params, real_images
        )["presence_probs"]

        _conditioned_model = handlers.condition(_bm, data={
            "observation_noise": _fixed_noise,
            "presence": _soft_presence,
        })
        # _frozen_guide = handlers.block(
        #     handlers.condition(
        #         handlers.substitute(
        #             active_version.guide,
        #             data={active_version.extras["guide_param_name"]: trained_guide_params},
        #         ),
        #         data={"observation_noise": _fixed_noise, "presence": _soft_presence},
        #     ),
        #     hide=[active_version.extras["guide_param_name"]],
        # )
        _frozen_guide = handlers.condition(
                active_version.guide,
                data={"observation_noise": _fixed_noise, "presence": _soft_presence},
        )

        from numpyro.infer import SVI, TraceMeanField_ELBO
        import numpyro.optim as _nopt
        _svi = SVI(_conditioned_model, _frozen_guide, _nopt.Adam(_svi_lr), TraceMeanField_ELBO())
        _svi_result = _svi.run(random.PRNGKey(_svi_seed), _svi_steps, real_images, progress_bar=False)
        trained_renderer_params = _svi_result.params[active_version.extras["renderer_param_name"]]
        svi_losses = np.asarray(_svi_result.losses)

        # Quick reconstruction metric (point-estimate render).
        _est_real = active_version.guide_point_estimates(trained_guide_params, real_images)
        _init_render = active_version.render(_est_real)
        _final_render = active_version.render(_est_real, renderer_params=trained_renderer_params)
        _mse_init = float(jnp.mean((real_images - _init_render) ** 2))
        _mse_final = float(jnp.mean((real_images - _final_render) ** 2))

        svi_summary = {
            "steps": _svi_steps,
            "lr": _svi_lr,
            "sigma_obs": _svi_sigma,
            "loss_first": float(svi_losses[0]),
            "loss_last": float(svi_losses[-1]),
            "loss_min": float(np.min(svi_losses)),
            "recon_mse_init": _mse_init,
            "recon_mse_after_svi": _mse_final,
            "recon_mse_ratio": _mse_final / _mse_init,
        }
    else:
        trained_renderer_params = None
        svi_losses = None
        svi_summary = {"SVI": "skipped: no trained_guide_params"}

    svi_summary

    return svi_losses, svi_summary, trained_renderer_params


@app.cell(hide_code=True)
def _(
    active_version,
    add_pred_circles,
    np,
    plt,
    real_estimates_pre_sc,
    real_fit_pre_sc,
    real_images,
    show_img,
    svi_summary,
    trained_renderer_params,
):
    if trained_renderer_params is not None:
        real_fit_post_svi = active_version.render(
            real_estimates_pre_sc, renderer_params=trained_renderer_params
        )
        real_residual_post_svi = np.abs(np.asarray(real_images) - np.asarray(real_fit_post_svi))
        # Cap displayed rows; the full batch is huge and chokes the marimo cell output.
        _max_rows = 8
        _rows_post = min(_max_rows, real_images.shape[0])
        fig_real_fit_post_svi, axes_post = plt.subplots(_rows_post, 4, figsize=(8.5, 1.9 * _rows_post), dpi=90)
        axes_post = np.asarray(axes_post).reshape(_rows_post, 4)
        for _row in range(_rows_post):
            panels = [
                (real_images[_row], "real + inferred slots"),
                (real_fit_pre_sc[_row], "initial renderer"),
                (real_fit_post_svi[_row], "SVI-trained renderer"),
                (real_residual_post_svi[_row], "|residual| (post-SVI)"),
            ]
            for _col, (_image, _title) in enumerate(panels):
                show_img(axes_post[_row, _col], _image)
                axes_post[_row, _col].set_title(_title if _row == 0 else "", fontsize=9)
            # Overlay inferred slots on the real image and on both renderer panels.
            for _col in (0, 1, 2):
                add_pred_circles(
                    axes_post[_row, _col],
                    active_version,
                    np.asarray(real_estimates_pre_sc["presence_probs"][_row]),
                    np.asarray(real_estimates_pre_sc["position"][_row]),
                    np.asarray(real_estimates_pre_sc["size"][_row]),
                )
        fig_real_fit_post_svi.suptitle(
            f"Renderer before vs. after SVI ({svi_summary['steps']} steps, "
            f"MSE {svi_summary['recon_mse_init']:.4f} -> {svi_summary['recon_mse_after_svi']:.4f})",
            y=0.998,
        )
        fig_real_fit_post_svi.tight_layout()
    else:
        fig_real_fit_post_svi = None
    fig_real_fit_post_svi

    return


@app.cell
def _(plt, svi_losses):
    plt.plot(svi_losses)
    return


@app.cell
def _(handlers, jax, jnp, optax, random):
    def sc_components(version, ratio_norm=None):
        if ratio_norm is None:
            ratio_norm = version.image_shape[0] * version.image_shape[1] * version.channels
        sc_chunk = int(version.extras.get("sc_chunk", 64))

        def sample_proposals(guide_params, images, n_samples, key):
            def draw(one_key):
                trace = handlers.trace(
                    handlers.seed(lambda batch: version.guide(batch, guide_params), one_key)
                ).get_trace(images)
                return {name: jax.lax.stop_gradient(trace[name]["value"]) for name in version.site_names}

            samples = [draw(k) for k in random.split(key, n_samples)]
            return {name: jnp.stack([sample[name] for sample in samples], axis=0) for name in version.site_names}

        def flatten_samples(samples):
            num_samples, batch = samples[version.site_names[0]].shape[:2]
            return num_samples, batch, {name: value.reshape((num_samples * batch,) + value.shape[2:]) for name, value in samples.items()}

        def broadcast_images(images, num_samples):
            return jnp.broadcast_to(images[None], (num_samples,) + images.shape).reshape((num_samples * images.shape[0],) + images.shape[1:])

        def chunked(fn, total, chunk):
            outs = []
            for start in range(0, total, chunk):
                end = min(start + chunk, total)
                outs.append(fn(start, end))
            return jnp.concatenate(outs, axis=0)

        def log_joint_samples(images, samples):
            num_samples, batch, flat = flatten_samples(samples)
            flat_images = broadcast_images(images, num_samples)
            values = chunked(
                lambda start, end: version.model_log_joint(
                    flat_images[start:end], {name: value[start:end] for name, value in flat.items()}
                ),
                flat_images.shape[0],
                sc_chunk,
            )
            return values.reshape((num_samples, batch))

        def guide_log_prob_samples(guide_params, images, samples):
            num_samples, batch, flat = flatten_samples(samples)
            flat_images = broadcast_images(images, num_samples)
            values = chunked(
                lambda start, end: version.guide_log_prob(
                    guide_params, flat_images[start:end], {name: value[start:end] for name, value in flat.items()}
                ),
                flat_images.shape[0],
                max(sc_chunk, 8),
            )
            return values.reshape((num_samples, batch))

        def sc_loss(guide_params, images, frozen_samples, frozen_log_joint):
            log_q = guide_log_prob_samples(guide_params, images, frozen_samples)
            ratio = (frozen_log_joint - log_q) / ratio_norm
            return jnp.mean(jnp.var(ratio, axis=0))

        return {
            "sample_proposals": sample_proposals,
            "log_joint_samples": log_joint_samples,
            "guide_log_prob_samples": guide_log_prob_samples,
            "sc_loss": sc_loss,
        }

    def make_sc_trainer(sc, lr=2e-5, clip=10.0):
        optimizer = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))

        @jax.jit
        def step(guide_params, opt_state, images, frozen_samples, frozen_log_joint):
            loss_value, gradients = jax.value_and_grad(sc["sc_loss"])(
                guide_params, images, frozen_samples, frozen_log_joint
            )
            updates, opt_state = optimizer.update(gradients, opt_state, guide_params)
            return optax.apply_updates(guide_params, updates), opt_state, loss_value, optax.global_norm(gradients)

        return optimizer, step

    return make_sc_trainer, sc_components


@app.cell
def _(
    active_version,
    make_sc_trainer,
    random,
    real_images,
    run_config,
    sc_components,
    tqdm,
    trained_guide_params,
):
    if run_config.run_sc and trained_guide_params is not None and run_config.sc_steps > 0:
        sc = sc_components(active_version)
        opt_sc, step_sc = make_sc_trainer(sc, lr=2e-5, clip=1.0)
        guide_params_sc = trained_guide_params
        opt_state_sc = opt_sc.init(guide_params_sc)
        fixed_samples = sc["sample_proposals"](
            guide_params_sc, real_images, run_config.sc_fixed_proposals, random.PRNGKey(500)
        )
        fixed_log_joint = sc["log_joint_samples"](real_images, fixed_samples)
        sc_before = float(sc["sc_loss"](guide_params_sc, real_images, fixed_samples, fixed_log_joint))
        sc_history = []
        key = random.PRNGKey(501)
        for step_id in tqdm(range(run_config.sc_steps)):
            key, subkey = random.split(key)
            proposals = sc["sample_proposals"](guide_params_sc, real_images, run_config.sc_proposals, subkey)
            proposal_log_joint = sc["log_joint_samples"](real_images, proposals)
            guide_params_sc, opt_state_sc, loss_value, grad_norm = step_sc(
                guide_params_sc, opt_state_sc, real_images, proposals, proposal_log_joint
            )
            if step_id % 10 == 0 or step_id == run_config.sc_steps - 1:
                fixed_value = float(sc["sc_loss"](guide_params_sc, real_images, fixed_samples, fixed_log_joint))
                sc_history.append((step_id, float(loss_value), float(grad_norm), fixed_value))
        sc_after = float(sc["sc_loss"](guide_params_sc, real_images, fixed_samples, fixed_log_joint))
        sc_summary = {"before": sc_before, "after": sc_after, "history": sc_history}
    else:
        guide_params_sc = None
        sc_summary = {"SC": "disabled; enable the Run real-image SC widget and set SC steps > 0"}
    sc_summary
    return (guide_params_sc,)


@app.cell
def _(active_version, guide_params_sc, np, plt, real_images, show_img):
    if guide_params_sc is not None:
        estimates_sc = active_version.guide_point_estimates(guide_params_sc, real_images)
        fit_sc = active_version.render(estimates_sc)
        residual_sc = np.abs(np.asarray(real_images) - np.asarray(fit_sc))
        _rows_sc = real_images.shape[0]
        fig_sc_fit, axes_sc_fit = plt.subplots(_rows_sc, 3, figsize=(8, 2.35 * _rows_sc))
        axes_sc_fit = np.asarray(axes_sc_fit).reshape(_rows_sc, 3)
        for _row in range(_rows_sc):
            for _col, (_image, _title) in enumerate([(real_images[_row], "real"), (fit_sc[_row], "SC fit"), (residual_sc[_row], "|residual|")]):
                show_img(axes_sc_fit[_row, _col], _image)
                axes_sc_fit[_row, _col].set_title(_title if _row == 0 else "", fontsize=9)
        fig_sc_fit.suptitle("Self-consistency fit to real patches", y=0.998)
        fig_sc_fit.tight_layout()
    else:
        fig_sc_fit = None
    fig_sc_fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Run summary and memory profile

    "
        f"Training summary: `{training_summary}`

    "
        f"Real patch summary: `{real_summary}`

    "
        "Memory log entries are `(label, max_rss_mib, unix_time_rounded)`:  "
        f"`{memory_log.records}`
    """)
    return


if __name__ == "__main__":
    app.run()
