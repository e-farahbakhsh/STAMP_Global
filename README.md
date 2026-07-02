# Spatiotemporal prospectivity modelling of porphyry mineralisation on a global scale

The current workflow is compatible with pyGPlates v1.0.0 and GPlately v2.0.0 and is designed to work with any plate reconstruction model available in the Plate Model Manager. To set up a suitable environment for running the workflow, please use the following commands:

```bash
conda install python=3.12 pip git notebook
pip install git+https://github.com/pulearn/pulearn.git@master
pip install gplately
pip install slabdip
pip install git+https://github.com/brmather/melt.git@main
pip install ipywidgets cmcrameri moviepy rioxarray scikit-image seaborn scikit-optimize shap
