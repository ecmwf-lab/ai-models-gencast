# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import dataclasses
import functools
import gc
import logging
import os

import numpy as np
import xarray
from ai_models.model import Model

from .input import create_training_xarray
from .output import save_output_xarray

LOG = logging.getLogger(__name__)


try:
    import haiku as hk
    import jax
    from graphcast import checkpoint
    from graphcast import data_utils
    from graphcast import gencast
    from graphcast import nan_cleaning
    from graphcast import normalization
    from graphcast import rollout
    from graphcast import xarray_jax

except ModuleNotFoundError as e:
    msg = "You need to install Graphcast/Gencast from git to use this model. See README.md for details."
    LOG.error(msg)
    raise ModuleNotFoundError(f"{msg}\n{e}")


class GenCast(Model):
    download_url = "https://storage.googleapis.com/dm_graphcast/gencast/{file}"
    expver = "dmge"

    grib_edition = 1
    grib_extra_metadata = {"type": "pf", "stream": "enfo"}

    # Download
    download_files = [
        "stats/diffs_stddev_by_level.nc",
        "stats/mean_by_level.nc",
        "stats/stddev_by_level.nc",
        "stats/min_by_level.nc",
    ]

    # Input
    area = [90, 0, -90, 360]

    param_sfc = [
        "lsm",
        "2t",
        "sst",
        "msl",
        "10u",
        "10v",
        "tp",
        "z",
    ]

    param_level_pl = (
        ["t", "z", "u", "v", "w", "q"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    forcing_variables = [
        "toa_incident_solar_radiation",
        # Not calling julian day and day here, due to differing assumptions with Deepmind
        # Those forcings are created by graphcast.data_utils
    ]

    use_an = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hour_steps = 12
        self.lagged = [-12, 0]
        self.params = None
        self.ordering = self.param_sfc + [
            f"{param}{level}" for param in self.param_level_pl[0] for level in self.param_level_pl[1]
        ]

        if isinstance(self.member_number, str):
            self.member_number = list(set(map(int, self.member_number.split(","))))
        elif isinstance(self.member_number, int):
            self.member_number = [int(self.member_number)]
        elif self.member_number is None:
            self.member_number = list(range(1, self.num_ensemble_members + 1))
        else:
            raise TypeError(f"`member_number` must be a string or int, not {type(self.member_number)}")

        if not len(self.member_number) == self.num_ensemble_members:
            raise ValueError(
                f"Number of ensemble members must match `member_number`,\nNot {self.num_ensemble_members=} and {self.member_number=}"
            )

    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def _with_configs(self, fn):
        return functools.partial(
            fn,
            sampler_config=self.sampler_config,
            task_config=self.task_config,
            denoiser_architecture_config=self.denoiser_architecture_config,
            noise_config=self.noise_config,
            noise_encoder_config=self.noise_encoder_config,
        )

    # Always pass params and state, so the usage below are simpler
    def _with_params(self, fn):
        return functools.partial(fn, params=self.params, state=self.state)

    # Deepmind models aren't stateful, so the state is always empty, so just return the
    # predictions. This is required by the rollout code, and generally simpler.
    @staticmethod
    def _drop_state(fn):
        def internal_func(*a, **kw):
            ex_kw = {"rng": a[0], "inputs": a[1], "targets_template": a[2], "forcings": a[3]}
            return fn(**ex_kw, **kw)[0]

        return internal_func

    def load_model(self):
        with self.timer(f"Loading {self.download_files[-1]}"):

            def get_path(filename):
                return os.path.join(self.assets, filename)

            diffs_stddev_by_level = xarray.load_dataset(get_path(self.download_files[0])).compute()

            mean_by_level = xarray.load_dataset(get_path(self.download_files[1])).compute()

            stddev_by_level = xarray.load_dataset(get_path(self.download_files[2])).compute()

            min_by_level = xarray.load_dataset(get_path(self.download_files[3])).compute()

            def construct_wrapped_gencast(
                sampler_config, task_config, denoiser_architecture_config, noise_config, noise_encoder_config
            ):
                """Constructs and wraps the GenCast Predictor."""
                predictor = gencast.GenCast(
                    sampler_config=sampler_config,
                    task_config=task_config,
                    denoiser_architecture_config=denoiser_architecture_config,
                    noise_config=noise_config,
                    noise_encoder_config=noise_encoder_config,
                )

                predictor = normalization.InputsAndResiduals(
                    predictor,
                    diffs_stddev_by_level=diffs_stddev_by_level,
                    mean_by_level=mean_by_level,
                    stddev_by_level=stddev_by_level,
                )

                predictor = nan_cleaning.NaNCleaner(
                    predictor=predictor,
                    reintroduce_nans=True,
                    fill_value=min_by_level,
                    var_to_clean="sea_surface_temperature",
                )

                return predictor

            @hk.transform_with_state
            def run_forward(
                inputs,
                targets_template,
                forcings,
                *,
                sampler_config,
                task_config,
                denoiser_architecture_config,
                noise_config,
                noise_encoder_config,
            ):
                predictor = construct_wrapped_gencast(
                    sampler_config, task_config, denoiser_architecture_config, noise_config, noise_encoder_config
                )
                return predictor(
                    inputs,
                    targets_template=targets_template,
                    forcings=forcings,
                )

            with open(get_path(self.download_files[-1]), "rb") as f:
                self.ckpt = checkpoint.load(f, gencast.CheckPoint)
                self.params = self.ckpt.params
                self.state = {}

                self.task_config = self.ckpt.task_config
                self.sampler_config = self.ckpt.sampler_config
                self.noise_config = self.ckpt.noise_config
                self.noise_encoder_config = self.ckpt.noise_encoder_config
                self.denoiser_architecture_config = self.ckpt.denoiser_architecture_config

                # Replace attention mechanism.
                # See https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md#running-inference-on-gpu
                splash_spt_cfg = self.ckpt.denoiser_architecture_config.sparse_transformer_config
                tbd_spt_cfg = dataclasses.replace(splash_spt_cfg, attention_type="triblockdiag_mha", mask_type="full")
                self.denoiser_architecture_config = dataclasses.replace(
                    self.ckpt.denoiser_architecture_config, sparse_transformer_config=tbd_spt_cfg
                )

                LOG.info("Model description: %s", self.ckpt.description)
                LOG.info("Model license: %s", self.ckpt.license)

            jax.jit(self._with_configs(run_forward.init))

            self.model = xarray_jax.pmap(
                jax.jit(self._with_params(self._with_configs(self._drop_state(run_forward.apply)))), dim="sample"
            )

    def run(self):

        oper_fcst: bool = False
        if self.num_ensemble_members == 0:
            oper_fcst = True
            # Set the number of ensemble members to 1, and id to 0.
            self.num_ensemble_members = 1
            self.member_number = [0]
            self.grib_extra_metadata = {"type": "fc", "stream": "oper"}

        if not (self.num_ensemble_members % len(jax.local_devices())) == 0:
            raise ValueError(
                f"Number of ensemble members must be divisible by number of devices, not {self.num_ensemble_members} and {len(jax.local_devices())}"
            )

        # Remove extra metadata to save input fields
        _metadata = self.grib_extra_metadata
        self.grib_extra_metadata = {}
        # We ignore 'tp' so that we make sure that step 0 is a field of zero values
        self.write_input_fields(self.fields_sfc, ignore=["tp"], accumulations=["tp"])
        self.write_input_fields(self.fields_pl)
        self.grib_extra_metadata = _metadata

        with self.timer("Building model"):
            self.load_model()

        with self.timer("Creating input data (total)"):
            with self.timer("Creating training data"):
                training_xarray, time_deltas = create_training_xarray(
                    fields_sfc=self.fields_sfc,
                    fields_pl=self.fields_pl,
                    lagged=self.lagged,
                    start_date=self.start_datetime,
                    hour_steps=self.hour_steps,
                    lead_time=self.lead_time,
                    forcing_variables=self.forcing_variables,
                    constants=self.override_constants,
                    timer=self.timer,
                )

            gc.collect()

            if self.debug:
                training_xarray.to_netcdf("training_xarray.nc")

            with self.timer("Extracting input targets"):
                (
                    input_xr,
                    template,
                    forcings,
                ) = data_utils.extract_inputs_targets_forcings(
                    training_xarray,
                    target_lead_times=[
                        f"{int(delta.days * 24 + delta.seconds/3600):d}h" for delta in time_deltas[len(self.lagged) :]
                    ],
                    **dataclasses.asdict(self.task_config),
                )

            if self.debug:
                input_xr.to_netcdf("input_xr.nc")
                forcings.to_netcdf("forcings_xr.nc")

        with self.timer("Doing full rollout prediction in JAX"):
            rng = jax.random.PRNGKey(0)
            # Fold in the member number to the random key
            rngs = np.stack([jax.random.fold_in(rng, i) for i in self.member_number], axis=0)

            chunks = []
            for chunk in rollout.chunked_prediction_generator_multiple_runs(
                self.model,
                rngs=rngs,
                inputs=input_xr,
                targets_template=template * np.nan,
                forcings=forcings,
                num_steps_per_chunk=1,
                num_samples=self.num_ensemble_members,
                pmap_devices=jax.local_devices(),
            ):
                chunks.append(chunk)
            output = xarray.combine_by_coords(chunks)

            if self.debug:
                output.to_netcdf("output.nc")

        with self.timer("Saving output data"):
            save_output_xarray(
                output=output,
                write=self.write,
                target_variables=self.task_config.target_variables,
                all_fields=self.all_fields,
                ordering=self.ordering,
                lead_time=self.lead_time,
                hour_steps=self.hour_steps,
                num_ensemble_members=self.num_ensemble_members,
                lagged=self.lagged,
                oper_fcst=oper_fcst,
                member_number=self.member_number,
            )

    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") not in ("an", "fc"):
            return

        if r.get("stream", "oper") not in ("oper", "scda"):
            return

        if self.use_an:
            r["type"] = "an"
        else:
            r["type"] = "fc"

        time = r.get("time", 12)

        r["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]

    def parse_model_args(self, args):
        import argparse

        parser = argparse.ArgumentParser("ai-models gencast")
        parser.add_argument("--num-ensemble-members", type=int, help="Number of ensemble members to run", default=1)
        parser.add_argument(
            "--member-number", help="Member Number, can only be used if `num_ensemble_members` == 1", default=None
        )
        parser.add_argument("--use-an", action="store_true")
        parser.add_argument("--override-constants")
        return parser.parse_args(args)


class GenCast0p25deg(GenCast):
    grid = [0.25, 0.25]

    def __init__(self, *args, **kwargs):
        self.download_files.append("params/GenCast 0p25deg <2019.npz")
        super().__init__(*args, **kwargs)


class GenCast0p25degOper(GenCast0p25deg):
    def __init__(self, *args, **kwargs):
        self.download_files.append("params/GenCast 0p25deg Operational <2022.npz")
        super().__init__(*args, **kwargs)


class GenCast1p0deg(GenCast):
    grid = [1.0, 1.0]

    def __init__(self, *args, **kwargs):
        self.download_files.append("params/GenCast 1p0deg <2019.npz")
        super().__init__(*args, **kwargs)


class GenCast1p0degMini(GenCast1p0deg):
    def __init__(self, *args, **kwargs):
        self.download_files.append("params/GenCast 1p0deg Mini <2019.npz")
        super().__init__(*args, **kwargs)


def model(model_version, **kwargs):

    # select with --model-version

    models = {
        "0.25": GenCast0p25deg,
        "0.25-oper": GenCast0p25degOper,
        "1.0": GenCast1p0deg,
        "1.0-mini": GenCast1p0degMini,
        "mini": GenCast1p0degMini,
        "default": GenCast0p25deg,
        "latest": GenCast0p25deg,
    }

    if model_version not in models:
        LOG.error(f"Model version {model_version} not found, using default")
        LOG.error(f"Available models: {list(models.keys())}")
        raise ValueError(f"Model version {model_version} not found")

    return models[model_version](**kwargs)
