# CS5228-Team 404
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)](https://pypi.org/project/autogluon/)

In this project, we look into the market for condominiums in Singapore. We aim to predict the sale prices through data mining techniques, different machine learning models and AutoML frameworks. [Github Link](https://github.com/YanZehong/CS5228-404)

## Table of Codes
- EDA:  
    * `eda_plot.py`  
- Preprocess:  
    * `run_preprocess.ipynb`  
- Feature Importance Analysis:  
    * `aspects_fi.ipynb`  
- Baselines: LinearNN, Lasso, ElasticNet, Ridge, SVM, RF, GBoost, XGB, LightGBM, Stacked  
    * `run_baselines.ipynb`  
- Autogluon-Tabular:  
    * `run_autogluon.ipynb`  

## Project Set Up

This is a list of all requirements used in this project.

### Requirements for baselines

```
conda create -n cs5228-404 -y python=3.8 pip
conda activate cs5228-404
pip install jupyter d2l torch torchvision # basic setup
jupyter notebook # run_preprocess.ipynb, run_baselines.ipynb
```

To install requirements, run `pip install -r requirements.txt`.

### Requirements for EDA images plot

```
pip install numpy pandas seaborn scipy plotly
pip install -U matplotlib
pip install -U kaleido
python eda_plot.py
jupyter notebook # aspects_fi.ipynb for feature importance analysis
```

### Requirements for AutoGluon
Refer to [Install Instructions](https://auto.gluon.ai/stable/install.html) | Documentation ([Stable](https://auto.gluon.ai/stable/index.html) | [Latest](https://auto.gluon.ai/dev/index.html)) for details.

To install AutoGluon on Windows, it is recommended to use Anaconda:
```
conda create -n ag python=3.9 -y
source deactivate
conda activate ag
pip3 install -U pip
pip3 install -U setuptools wheel
# CPU version of pytorch has smaller footprint - see installation instructions in
# pytorch documentation - https://pytorch.org/get-started/locally/
pip3 install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip3 install --pre autogluon
```

For Mac users, if you don’t have them, please first install: XCode, Homebrew, LibOMP.

```
# Once you have Homebrew, LibOMP can be installed via:
brew install wget
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
# Uninstall libomp if it was previous installed
# brew uninstall libomp
brew install libomp.rb
rm libomp.rb

conda create -n ag python=3.9 -y
source deactivate
conda activate ag
pip3 install -U pip
pip3 install -U setuptools wheel
# CPU version of pytorch has smaller footprint - see installation instructions in
# pytorch documentation - https://pytorch.org/get-started/locally/
pip3 install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip3 install --pre autogluon
```

Then, run the notebook scripts:
```
jupyter notebook # run_autogluon.ipynb
```

## Framework Overview
![preprocess](https://github.com/YanZehong/CS5228-404/blob/main/images/flowchart.png?raw=true)
![model-framework](https://github.com/YanZehong/CS5228-404/blob/main/images/model_framework.png?raw=true)

## Acknowledgement
We would like to thank Chris for helpful comments and feedback on earlier versions of this work. We are grateful to CS5228 for giving us such a valuable data mining experience.  
Team 404: Yan Zehong, Jiang Chuanqi, Li Xuanman, Gu Ruijia.  
[Kaggle Competition Link-CS5228-2021 (Semester 2) Final Project](https://www.kaggle.com/competitions/cs5228-2021-semester-2-final-project)  