# ai-models-gencast
[![Upload Python Package](https://github.com/ecmwf-lab/ai-models-gencast/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ecmwf-lab/ai-models-gencast/actions/workflows/python-publish.yml)

`ai-models-gencast` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run Google Deepmind's [GenCast](https://github.com/deepmind/graphcast).

GenCast: Diffusion-based ensemble forecasting for medium-range weather, arXiv preprint: 2312.15796, 2024. https://arxiv.org/abs/2312.15796

GenCast was created by Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R. Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, Remi Lam, Matthew Willson

The model weights are made available for use under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). You may obtain a copy of the License at: https://creativecommons.org/licenses/by-nc-sa/4.0/.

## Installation

To install the package, run:

```bash
pip install ai-models-gencast
```

This will install the package and most of its dependencies.

Then to install gencast dependencies (and Jax on GPU):

> [!CAUTION]
> GenCast requires significant GPU & Memory Resources.
> See [here](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md#gencast-memory-requirements)


### GenCast and Jax

GenCast depends on Jax, which needs special installation instructions for your specific hardware.

Please see the [installation guide](https://github.com/google/jax#installation) to follow the correct instructions.

We have prepared two `requirements.txt` you can use. A CPU and a GPU version:

For the preferred GPU usage:
```
pip install -r requirements-gpu.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For the slower CPU usage:
```
pip install -r requirements.txt
```

## Specifying ensemble numbers

There are three ways to control the ensemble members and behaviour of the `GenCast` `ai-model`.

| Description | Args | Result |
| ----------- | ---- | ------ |
| `type=fc`, single member | `--num-ensemble-members 0` | Will create a `grib` file of `type=fc` |
| N members per process with ID = `range(num-ensemble-members)` | `--num-ensemble-members $N>1` | N ensemble members created all in same process, with id from the range|
| N members per process with controlled ID | `--num-ensemble-members $N>1` `--member-number 1,2...N` | N ensemble members created all in same process, with id controlled from `member-number` |

With these approaches it is possible to create either a single forecast, many ensembles in a single process, or many ensembles over many processes.
