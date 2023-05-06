# Large-Scale Study of Temporal Shift in Health Insurance Claims

To reproduce the experiments in this paper:
1. Modify `config.py` to set the database name, schema, and output directories.
2. The `data_extraction` directory contains a pipeline to extract data for the large-scale study.
3. The `temporal_shift_scan` directory implements our algorithms to test and scan for temporal shift.

The `utils` directory contains supporting functions. To reproduce our conda environment, run `conda create --prefix NEWENV --file conda_env_pkgs.txt`.