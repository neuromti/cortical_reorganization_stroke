# Stimulation-Evoked Cortical Responses in Severe Chronic Stroke

Analysis scripts for: Stimulation-evoked cortical responses track sensorimotor reorganization in severe chronic stroke.

“This repository contains analysis scripts without patient-level data.”

## Repository Layout

- `Preprocess.py`: preprocessing pipeline for `.xdf` recordings. Includes marker correction, channel interpolation, epoching, filtering, autoreject, ICA cleaning, and figure/export steps.
- `meta_functions_HC.py`: reusable analysis and plotting functions shared across workflows (cluster-based stats, ERP component extraction, topomaps, laterality indices, and clinical correlation plots).
- `Main_file.py`: main analysis workflow for group-level summaries, ERP figures, cluster analyses, latency/amplitude metrics, and statistical plots.
- `requirements.txt`: Python package dependencies used by the scripts.

## Data Placement

Patient-level raw and derived data are intentionally excluded from this repository.

To run the scripts locally, place your data in the folder paths expected in the code (for example paths under `/home/user/data/...`) or update the path variables inside the scripts:

- `Preprocess.py`:
  - `exdir` for raw `.xdf` input
  - `save_folder` / `save_folder_figs` for output epochs and figures
- `Main_file.py`:
  - `epochs_data_dir` for preprocessed epochs
  - `Save_folder`, `save_folder`, and `save_folder_peak` for analysis outputs
  - additional references to intermediate files (e.g., pickled evoked data and clinical spreadsheet)

For portability after upload to GitHub, consider changing hard-coded absolute paths to project-relative paths.

## Package Dependencies

Core Python dependencies are listed in `requirements.txt` and include:

- `mne`
- `pyxdf`
- `pycircstat`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `lmfit`
- `tqdm`
- `autoreject`

Typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

- Scripts reflect the analysis workflow used for manuscript preparation.
- File paths and some subject-specific handling are currently encoded directly in scripts and should be adapted to your local environment/data organization.
