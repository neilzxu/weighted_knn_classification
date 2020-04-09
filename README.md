# Code for _Multiclass Classification via Class-Weighted Nearest Neighbors_
## Setup

Requires [conda](https://docs.conda.io/en/latest/) with Python 3.7.

1. Install dependencies with `conda env create -f environment.yml`
2. Download dataset with `./download_uci_data.sh`

## Experiment scripts

Before running experiments, make sure the conda environment is active by running `source activate wknn` or `conda activate wknn`.

There are 3 experiments scripts:
- The figures in section 5 showing convergence of the confusion matrix: `python knn_multiclass_example.py`.
- The figures in section 6 for synthetic data are plotted in the Jupyter notebook `knn.ipynb`.
- The results in section 6 for the real data: `real_exp.sh`

Results of these scripts will appear in the `results` directory.
