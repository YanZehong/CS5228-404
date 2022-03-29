# CS5228-404
In this project, we look into the market for condominiums in Singapore. We aim to predict the sale prices through data mining techniques, different machine learning models and AutoML frameworks.

## Project Set Up
This is a list of all requirements used in this project.
### Requirements for baselines

```
conda create -n cs5228-404 -y python=3.8 pip
conda activate cs5228-404
pip install jupyter d2l torch torchvision
jupyter notebook
```

### Requirements for EDA images plot 

```
pip install numpy pandas seaborn scipy plotly
pip install -U matplotlib
pip install -U kaleido
python eda_plot.py
```

### Requirements for AutoGluon
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

## Acknowledgement