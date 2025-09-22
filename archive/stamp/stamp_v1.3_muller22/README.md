The current workflow is compatible with pyGplates v1.0.0 and GPlately v2.0.0rc0. To set up a suitable environment for running the workflow, please use the following commands:

```bash
conda install python=3.12 pip git notebook
pip install git+https://github.com/pulearn/pulearn.git@master
pip install gplately==2.0.0rc0
pip install slabdip
pip install ipywidgets cmcrameri moviepy rioxarray scikit-image seaborn scikit-optimize shap
