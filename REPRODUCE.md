# Scope
This guide outlines the steps required to reproduce the results and figures of the thesis that were derived from this codebase.
For the BEELINE specific figures and results see reproduce.md at https://github.com/Nannich/Beeline.

## 1. Environment Setup

This codebase was developed and tested using Python 3.14. Ensure you have the correct Python version installed, then isolate the installation using a virtual environment.

```bash
git clone [https://github.com/Nannich/bachelor-thesis-kan-grn.git](https://github.com/Nannich/bachelor-thesis-kan-grn.git)
cd bachelor-thesis-kan-grn

python3.14 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Data Acquisition
1. Download the BEELINE datasets from [Zenodo (Record 7682713)](https://zenodo.org/records/7682713).
2. Extract the downloaded Curated and Synthetic dataset folders directly into the data/raw/ directory.
3. The raw Synthetic datasets include dropout variants (q50, q70) and various cell counts (100, 200, 500, 2000, 5000). Only keep the no-dropout, 2000-cell variants. After cleaning up the unzipped directories, your data/ directory structure should match the layout below:

```text
/bachelor-thesis-kan-grn$ tree
data
└── raw
    ├── Curated
    │   ├── GSD
    │   │   ├── GroundTruthNetwork.csv
    │   │   ├── GSD-2000-1
    │   │   │   ├── ExpressionData.csv
    │   │   │   └── PseudoTime.csv
    │   │   └── [GSD-2000-2 to GSD-2000-10 structured identically to GSD-2000-1]
    │   └── [HSC, mCAD, VSC folders structured identically to GSD]
    └── Synthetic
        ├── dyn-BF
        │   ├── GroundTruthNetwork.csv
        │   └── dyn-BF-2000
        │       ├── dyn-BF-2000-1
        │       │   ├── ExpressionData.csv
        │       │   ├── GroundTruthNetwork.csv
        │       │   └── PseudoTime.csv
        │       └── [dyn-BF-2000-2 to dyn-BF-2000-10 structured identically to dyn-BF-2000-1]
        └── [dyn-BFC, dyn-CY, dyn-LI, dyn-LL, dyn-TF folders structured identically to dyn-BF]

120 directories, 270 files
```

## 3. Figure and Table Mapping

The following table maps each figure and table presented in the thesis to its respective generation script and execution command. All generated outputs are saved under the `results/` directory. Before running the benchmarks delete data/raw/hESC-500-CellType/.

| Thesis Reference | Description | Execution Command |
| :--- | :--- | :--- |
| **Table 2 (App. 6-10)** | Trajectory benchmark results | `python main.py benchmark --mode trajectory data/raw` |
| **Figure 3 (App 15, 16)** | GRN inference configuration benchmark results | `python main.py benchmark --mode grn data/raw` |
| **Table 4** | Symbolic formula extraction benchmark results | `python main.py benchmark --mode symbolic data/raw/` |
| **Equation 10** | Deep symbolic equations for HSC-2000-1 | `python main.py symbolic extract HSC-2000-1 --prune` |
| **Figure 10a** | Scatter plot | `python main.py trajectory plot hESC-500-CellType --mode scatter --gene GATA4` |
| **Figure 10b** | Distribution plot | `python main.py trajectory plot hESC-500-CellType --mode distribution --gene GATA4` |
| **Figure 11a** | MSE trajectory | `python main.py trajectory train hESC-500-CellType --gene GATA4 --loss mse`, `python main.py trajectory plot hESC-500-CellType --mode trajectory --gene GATA4 --loss mse` |
| **Figure 11b** | ZINB trajectory | `python main.py trajectory train hESC-500-CellType --gene GATA4 --loss zinb --ridge_lambda 0.11`, `python main.py trajectory plot hESC-500-CellType --mode trajectory --gene GATA4 --loss zinb` |
| **Figure 12** | ZINB trajectory | `python main.py trajectory train HSC-2000-1 --gene Eklf --loss zinb`, `python main.py trajectory plot HSC-2000-1 --mode trajectory --gene Eklf --loss zinb`, `python main.py symbolic plot HSC-2000-1 --checkpoint models/HSC-2000-1/trajectory/kan_Eklf_zinb.pth` |
| **Table 11** | Symbolic equations for HSC-2000-1 | `python main.py symbolic extract HSC-2000-1 --prune --skip_deep` |

Reproduction instructions for figures 4 to 7 and 17 can be found at https://github.com/Nannich/Beeline.