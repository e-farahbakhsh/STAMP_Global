# Spatiotemporal prospectivity modelling of porphyry mineralisation on a global scale

This repository contains **STAMP** (Spatio-Temporal Analysis of Mineral Prospectivity), the open-source workflow used to build a global, full-Phanerozoic model of porphyry copper prospectivity that couples a machine learning formation probability model with a preservation likelihood framework derived from landscape evolution modelling. It accompanies the paper:

> Farahbakhsh, E., McInnes, B. I. A., Kohlmann, F., Seton, M., Dutkiewicz, A., Müller, R. D. (2026). *Global porphyry copper prospectivity through the Phanerozoic from interpretable machine learning coupling formation and preservation*. [Under review]

The workflow reconstructs subduction zone kinematics and downgoing plate properties through deep time and trains an interpretable classifier that combines positive–unlabelled bagging with a random forest. Across the global arc network and at one-million-year resolution, it estimates the probability that porphyry mineralisation formed, the uncertainty of that prediction, and the preservation likelihood that a deposit, once formed, has survived subsequent erosion, combining these into a map of preserved mineralisation probability. See the manuscript for full details.

The current workflow is compatible with pyGPlates v1.0.0 and GPlately v2.0.0 and is designed to work with any plate reconstruction model available in the Plate Model Manager.

## Workflow overview

STAMP is organised as six Jupyter notebooks, run in order, supported by a set of Python modules in the `lib/` package. The pipeline proceeds from feature extraction and reconstruction, through feature analysis and predictive modelling, to prospectivity mapping and preservation assessment.

| Notebook | Purpose |
| --- | --- |
| `01_feature_extraction.ipynb` | Sample trench points along the reconstructed subduction zones at each time step and extract the kinematic, downgoing plate (oceanic grid), and thermodynamically modelled slab devolatilisation features. |
| `02_reconstruction_coregistration.ipynb` | Build the one-sided trench buffer zones, reconstruct the known porphyry occurrences to their palaeo-positions, generate the unlabelled and target point sets, and coregister each point to its nearest trench point. |
| `03_feature_analysis.ipynb` | Prepare and downsample the data, examine the pairwise correlation structure and hierarchical clustering of the features, retain one representative feature per cluster, and generate provisional labels through positive–unlabelled bagging. |
| `04_predictive_modelling.ipynb` | Train the downstream random forest, evaluate it (ROC/AUC, calibration), interpret it (Gini and permutation importance, partial dependence and ICE, SHAP), and compute the mineralisation probability, uncertainty, and adjusted probability through time. |
| `05_prospectivity_map.ipynb` | Fill the present-day continental landmasses with a regular grid, reconstruct the grid through time within the trench buffer zones, predict the probability at each point, and produce the present-day global prospectivity maps with prediction–area (P–A) evaluation. |
| `06_preservation.ipynb` | Coregister the points with the cumulative erosion grids, remove outliers, fit a generalised gamma preservation likelihood function, and combine it with the formation probability to map the preserved mineralisation probability. |

### The `lib/` package

| Module | Role |
| --- | --- |
| `feature_extraction.py` | Trench tessellation and convergence kinematics, coregistration of oceanic grids, carbon, slab flux and cumulative subducted thickness calculations, and thermodynamic slab devolatilisation (H<sub>2</sub>O and CO<sub>2</sub> outflux) and arc magma flux. |
| `reconstruction_coregistration.py` | Trench buffer zones, reconstruction of the deposit, unlabelled and target point sets, nearest trench coregistration, direct sampling of crustal thickness, and imputation of missing values. |
| `slab_dip.py` | Wrapper around the `slabdip` predictor that returns slab dip and the associated arc–trench distance. |
| `water_thickness.py` | Downgoing plate water inventory, partitioned into sedimentary and crustal pore and bound water, mantle lithosphere hydration, and hydration at the base of the lithosphere. |
| `feature_analysis.py` | Downsampling with age and spatial balancing, and correlation analysis and reporting. |
| `predictive_modelling.py` | ROC/AUC plotting, entropy and tree-vote-variance uncertainty measures, and gridding and curve utilities. |
| `prospectivity_map.py` | Reconstruction of continental grid nodes filtered to the trench buffer zones through time. |
| `preservation.py` | Coregistration of points with the cumulative erosion grids and removal of outliers. |
| `plot.py` | Map and animation plotting (GPlately/Cartopy), including deposit overlays. |

The `lib/` folder must sit in the same directory as the notebooks, since each notebook imports from it (for example, `from lib.feature_extraction import *`).

## Installation

We recommend a dedicated conda environment:

```bash
conda create -n stamp python=3.12 pip git notebook
conda activate stamp
pip install git+https://github.com/pulearn/pulearn.git@master
pip install gplately
pip install slabdip
pip install git+https://github.com/brmather/melt.git@main
pip install ipywidgets cmcrameri moviepy rioxarray scikit-image seaborn scikit-optimize shap
```

Cartopy, NumPy, pandas, scikit-learn, SciPy, xarray, GeoPandas, and rasterio are installed as dependencies of the packages above.

## Configuration

Runtime settings are read from a `parameters.py` file in the repository root, which each notebook imports as `from parameters import parameters`. The `parameters` dictionary defines, among others:

- `plate_model` and `anchor_plate_id`: the Plate Model Manager model name and the anchor plate ID.
- `timespan` (`min`, `max`) and `temporal_resolution`: the reconstruction window and step, in Ma.
- `grid_resolution`: the spacing of the target and continental grid points, in degrees.
- `inputs_dir` and `outputs_dir`: the input data and output directories.
- The oceanic grid, slab H<sub>2</sub>O and CO<sub>2</sub>, and Perple_X lookup table sub-directories, together with output filenames.

Adjust these to point at your plate model and input data before running the notebooks.

## Input data

Beyond the plate reconstruction, which is fetched automatically through the Plate Model Manager, the workflow expects the following inputs under `inputs_dir` (sub-directory names follow those referenced in the notebooks):

- **Oceanic grids** reconstructed on the chosen plate model: seafloor age (`SeafloorAge`), spreading rate (`SpreadingRate`), total deep-sea sediment thickness (`SedimentThickness`), and upper-oceanic-crust carbon.
- **Perple_X lookup tables** for slab devolatilisation, organised by reservoir (`Sediments`, `Metabasalts`, `Intrusives`, `Sublithospheric_Oceanic_mantle`), each with H<sub>2</sub>O and CO<sub>2</sub> tables.
- **Crustal thickness grids**, used only by the secondary model.
- **Cumulative erosion grids** from a deep-time landscape evolution model, named `cumulative_erosion_{time}Ma.nc`, used to estimate preservation likelihood.
- **A global porphyry copper deposit database**, providing the positive (training) samples.

## Running the workflow

1. Prepare the environment and `parameters.py` as above, and place the input data under `inputs_dir`.
2. Run the notebooks in numerical order, `01` through `06`. Each notebook writes its outputs to `outputs_dir`, where they are picked up by the subsequent notebooks.
3. Present-day and time-dependent prospectivity maps, the preservation-conditioned maps, and the accompanying animations are produced by notebooks `04` to `06`.

## Citation

If you use this workflow, please cite the accompanying paper:

```bibtex
@article{Farahbakhsh2026,
  author  = {Farahbakhsh, E. and McInnes, B. I. A. and Kohlmann, F. and
             Seton, M. and Dutkiewicz, A. and M\"uller, R. D.},
  title   = {Global porphyry copper prospectivity through the {Phanerozoic}
             from interpretable machine learning coupling formation and
             preservation},
  journal = {???},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Ehsan Farahbakhsh — e.farahbakhsh@sydney.edu.au or ehsan.farahbakhsh@anu.edu.au
