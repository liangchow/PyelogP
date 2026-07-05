# PyelogP

[![PyPI](https://img.shields.io/pypi/v/pyelogp)](https://pypi.org/project/pyelogp/)

"Py e-log(P)" is a lightweight Python library for estimating the preconsolidation pressure (*P'c*) of oedometer test data using the strain-energy method with knee-point detection and thresholding in the e-log(P) space.

![San Francisco Bay Mud (Bonaparte & Mitchell, 1976)](https://raw.githubusercontent.com/liangchow/PyelogP/main/images/sfbaymud.png)<br/>
![Wallaceburg clay (Becker et al., 1987)](https://raw.githubusercontent.com/liangchow/PyelogP/main/images/wallaceburgclay.png)<br/>
![Louiseville clay (TPM, 1996)](https://raw.githubusercontent.com/liangchow/PyelogP/main/images/louisevilleclay.png)

## Features

- Manual data input or import from a `.csv` file.
- Automatic removal of unload–reload cycles from input data.
- (*Beta*) Threshold-based fitting range selection for S-shaped curves.

## Installation

```bash
pip install pyelogp
```

### Clone from GitHub

```bash
# Clone the repo
git clone https://github.com/liangchow/pyelogp.git
cd pyelogp

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -e .
```

## Quick start

See `run.py` for a full example, including loading data from a CSV file via `Data.from_csv`.

```python
from pyelogp import Data

# Louiseville clay
pressure = [59, 90, 120, 150, 165, 172, 184, 222, 300, 400]
void_ratio = [2.115, 2.113, 2.098, 2.083, 2.055, 2, 1.8, 1.5, 1.3, 1.193]
e0 = 1.21 (Optional)

d = Data(pressure=pressure, void_ratio=void_ratio, e0=e0)

# Run analysis
result = d.find_pc()
print(f"Pc  = {result.pc:.1f}")
print(f"e @ Pc = {result.e_pc:.4f}")
print(f"R²(seg1) = {result.r2_seg1:.4f}")
print(f"R²(seg2) = {result.r2_seg2:.4f}")

# Return
Preconsolidation pressure (Pc): 164.49
Void ratio at pc (e_pc): 2.0573
R² segment 1: 0.893768
R² segment 2: 0.96937
```

## Contributing

Contributions are welcome, please refer to [CONTRIBUTING](https://github.com/liangchow/pyelogp/blob/main/CONTRIBUTING.md)
to learn more about how to contribute.

## References

Becker, D. E., Crooks, J. H. A., Been, K., & Jefferies, M. G. (1987).
*Work as a Criterion for Determining in situ and Yield Stresses in Clays.*
Canadian Geotechnical Journal, 24(4), 549–564.

Bonaparte, R. and Mitchell, J.K. (1979).
*The Properties of San Francisco Bay Mud at Hamilton Air Force Base, California.* 
Geotechnical Engineering Report, University of California, Berkeley, April 1979.

Satopa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). 
*Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System Behavior.* 
31st International Conference on Distributed Computing Systems Workshops, 166-171.

Terzaghi, K., Peck, R.B., & Mesri, G. (1996). 
*Soil Mechanics in Engineering Practice (3rd ed.).* 
John Wiley & Sons, Inc.
