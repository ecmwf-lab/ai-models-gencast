# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict

import numpy as np

from .convert import GRIB_TO_CF
from .convert import GRIB_TO_XARRAY_PL
from .convert import GRIB_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)

ACCUMULATION_VALUES = defaultdict(lambda: defaultdict(lambda: 0))


def accumulate(values, param: str, ensemble_number: int):
    """Accumulate values for a given parameter and ensemble member"""
    ACCUMULATION_VALUES[param][ensemble_number] += values
    return ACCUMULATION_VALUES[param][ensemble_number]


def save_output_xarray(
    *,
    output,
    target_variables,
    write,
    all_fields,
    ordering,
    time,
    hour_steps,
    num_ensemble_members,
    lagged,
    oper_fcst,
    member_numbers,
):
    # LOG.info("Converting output xarray to GRIB and saving")

    # output["total_precipitation_12hr"] = output.data_vars["total_precipitation_12hr"].cumsum(dim="time")

    all_fields = all_fields.order_by(
        valid_datetime="descending",
        param_level=ordering,
        remapping={"param_level": "{param}{levelist}"},
    )

    for fs in all_fields[: len(all_fields) // len(lagged)]:
        param, level = fs.metadata("shortName"), fs.metadata("levelist", default=None)
        for i in range(num_ensemble_members):
            ensemble_member = member_numbers[i]  # Associated member number
            time_idx = 0  # As we are saving each time individually, the index to select is 0

            if level is not None:
                param = GRIB_TO_XARRAY_PL.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time_idx).sel(level=level).isel(sample=i).data_vars[param].values
            else:
                param = GRIB_TO_CF.get(param, param)
                param = GRIB_TO_XARRAY_SFC.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time_idx).isel(sample=i).data_vars[param].values

            # We want to field north=>south

            values = np.flipud(values.reshape(fs.shape))

            if oper_fcst:
                extra_write_kwargs = {}
            else:
                extra_write_kwargs = dict(number=ensemble_member)

            if param == "total_precipitation_12hr":
                values = accumulate(values, param, ensemble_member)
                write(
                    values, template=fs, stepType="accum", startStep=0, endStep=time * hour_steps, **extra_write_kwargs
                )
                # NOTE: stepType must be before startStep and endStep
            else:
                write(
                    values,
                    template=fs,
                    step=time * hour_steps,
                    **extra_write_kwargs,
                )
