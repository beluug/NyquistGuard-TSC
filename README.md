# NyquistGuard-TSC

NyquistGuard-TSC is a rate-explicit dual-path time-series classifier for discrete,
unseen sampling-rate shifts when sampling-rate metadata are available at inference.
This directory is the prepared public code and frozen-results package for version
V5.1.

## Scope of this package

- `src/nyquistguard/`: model, data preparation, experiment and analysis code.
- `configs/`: frozen model and experiment configurations used by the reported runs.
- `results/frozen/`: CSV tables, figures and provenance generated from frozen reports.
- `reports/`: the four-dataset independent confirmation, ten-dataset retrospective
  extension, efficiency benchmark and component-ablation reports.
- `docs/DATA_AND_REPRODUCTION.md`: source links, licence notes and reproduction
  boundaries for every dataset family.
- `tests/`: unit and integration tests that do not require redistributing source data.

The V5.1 experimental results are frozen. The ten-dataset extension reuses an
already-accessed 210-run baseline matrix and is retrospective; the independent
evidence is the four previously untouched datasets described in the reports. The
package does not claim universal improvement, independent state of the art, or
improved selective reliability. PAMAP2 degradation and the reliability boundary are
reported explicitly in the manuscript and reports.

The historical `configs/experiments/full.yaml` records the pre-amendment baseline
setting. The frozen Full report records and enforces the later 1,000-kernel
MultiROCKET resource amendment through `src/nyquistguard/experiments/full_parallel.py`;
the amendment identifier is retained in the report provenance.

## Installation

Use Python 3.10 or newer. The lock file records the tested dependency set; a CUDA
installation is optional for reading the frozen tables and running unit tests.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe -m pytest
```

## Reproduction

The repository contains code and metadata, not original data. Obtain each dataset
from its authoritative source and follow `docs/DATA_AND_REPRODUCTION.md`. Raw files,
processed signal caches, local run directories and private submission materials are
intentionally excluded. Re-running the full training/test matrix is not required to
use the frozen results and must not be interpreted as an independent confirmation.

The top-level `run_experiments.py` exposes the experiment stages. Stages that require
manual confirmation are guarded and do not start automatically. The reports in this
package are the canonical numerical source for the accompanying manuscript draft.

## Citation and licence

Please cite the accompanying NyquistGuard-TSC manuscript when a public article record
is available, and cite every original dataset according to its repository terms. The
code in this package is released under the MIT License. Dataset licences remain
separate and are not relicensed by this repository.

Public repository: https://github.com/beluug/NyquistGuard-TSC  
Current release commit: `6a10c7dc59f2e0424ee4cec82378a016e1cdcaed`  
Zenodo DOI: not yet available; add it here and in `CITATION.cff` after archival.
