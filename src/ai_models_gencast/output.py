# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

from .convert import GRIB_TO_CF
from .convert import GRIB_TO_XARRAY_PL
from .convert import GRIB_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)


def save_output_xarray(
    *,
    output,
    target_variables,
    write,
    all_fields,
    ordering,
    lead_time,
    hour_steps,
    num_ensemble_members,
    lagged,
    oper_fcst,
    member_number,
):
    LOG.info("Converting output xarray to GRIB and saving")

    output["total_precipitation_12hr"] = output.data_vars["total_precipitation_12hr"].cumsum(dim="time")

    all_fields = all_fields.order_by(
        valid_datetime="descending",
        param_level=ordering,
        remapping={"param_level": "{param}{levelist}"},
    )

    for time in range(lead_time // hour_steps):
        for fs in all_fields[: len(all_fields) // len(lagged)]:
            param, level = fs.metadata("shortName"), fs.metadata("levelist", default=None)
            for i in range(num_ensemble_members):
                ensemble_member = member_number[i]

                if level is not None:
                    param = GRIB_TO_XARRAY_PL.get(param, param)
                    if param not in target_variables:
                        continue
                    values = output.isel(time=time).sel(level=level).sel(sample=i).data_vars[param].values
                else:
                    param = GRIB_TO_CF.get(param, param)
                    param = GRIB_TO_XARRAY_SFC.get(param, param)
                    if param not in target_variables:
                        continue
                    values = output.isel(time=time).sel(sample=i).data_vars[param].values

                # We want to field north=>south

                values = np.flipud(values.reshape(fs.shape))

                if oper_fcst:
                    extra_write_kwargs = {}
                else:
                    extra_write_kwargs = dict(number=ensemble_member)

                if param == "total_precipitation_12hr":
                    write(values, template=fs, startStep=0, endStep=(time + 1) * hour_steps, **extra_write_kwargs)
                else:
                    write(
                        values,
                        template=fs,
                        step=(time + 1) * hour_steps,
                        **extra_write_kwargs,
                    )
