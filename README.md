[![Pre-release](https://img.shields.io/github/v/release/DmitriiGurev/genet-py?include_prereleases&label=pre-release&color=orange)](https://github.com/DmitriiGurev/genet-py/releases) [![PyPI](https://img.shields.io/badge/dynamic/json?label=PyPI&url=https://pypi.org/pypi/genet-py/json&query=$.info.version&color=blue)](https://pypi.org/project/genet-py/)

<p>
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/DmitriiGurev/genet-py/master/assets/logo-light.png">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/DmitriiGurev/genet-py/master/assets/logo-dark.png">
        <img src="https://raw.githubusercontent.com/DmitriiGurev/genet-py/master/assets/logo-dark.png" alt="genet-py logo" width="100">
    </picture>
</p>

# GENET

*GENET* is a global machine learning model of the near-Earth electron environment.

The model is a neural network that can reconstruct pitch-angle distributions of 0.1–100 keV electron fluxes at distances within ~20 Earth radii. Trained on Cluster observations.

## Installation

```bash
pip install genet-py
```

You need a valid [SuperMAG account](https://supermag.jhuapl.edu/indices/) to download the SME index.

## Usage

### Inputs

- `time`: UTC datetime
- `coords_gsm`: position in GSM coordinates, in Earth radii (RE)
- `energy`: electron energy in keV, from 0.1 to 80
- `pitch_angle`: pitch angle in degrees
- `percentile`: predicted flux percentile, one of 5, 25, 50, 75, 95. The default is 50 (median).

Each input parameter can be either a single value or a list of values.

### Output

- Electron flux in 1 / (cm2 s sr keV)

### Example

```python
from genet import GENET
from datetime import datetime

genet = GENET(supermag_username="your_supermag_username")

flux = genet.predict(
    time=datetime(2015, 3, 17),  # UTC
    coords_gsm=(6.6, 0, 0),      # GSM coordinates in RE
    energy=50,                   # Energy in keV
    pitch_angle=90,              # Pitch angle in degrees
    percentile=50                # 5, 25, 50, 75, or 95
)

# Flux in 1 / (cm2 s sr keV)
print(flux)
```