# instance_halos (ihs)
This repository contains the semantic and instance segmentation models presented in the paper "Characterizing structure formation through instance segmentation". A Jupyter notebook is included for generating example figures analogous to those presented in the paper.

## Software dependencies and datasets
This code uses `numpy`, `scipy`, `matplotlib`, `networkx`, and `cython` packages.

The data employed to train and test the networks has been generated using l-gadget3 code [^1] and using a private version of the `bacco` package[^2].

[^1]: <https://academic.oup.com/mnras/article-abstract/507/4/5869/6328503?redirectedFrom=fulltext&login=true>
[^2]: <https://bacco.dipc.org/index.html>

## Code description
The code is organized as follows:
    - The `utils_modules` folder contains the codes employed to load the dataset and the model. It also includes the codes employed for performing the pseudo-space clsutering, generate the model predictions, and carry out statistical analysis and plots.
    - The `example` folder contains an example Jupyter notebook (`example.ipynb`) to see how the models are able to generate predictions and plots similar to those presented in the paper.
    - After cloning the huggingface repositiores (see installation instructions below) the folder named `instance_halos_data` will contain .npy files used as example inputs for the networks (also the ground truth predictions). The `instance_halos_models` folder will contain the semantic and instance models stored in a format compatible with tensorflow.
	
## installation modules
```bash
conda create -n VE_IHs
conda activate VE_IHs
conda install -c anaconda tensorflow-gpu
conda install jupyter numpy scipy matplotlib networkx cython ipykernel
git clone --recursive git@github.com:daniellopezcano/instance_halos.git
cd instance_halos
pip install -e .
```
If the main repository is cloned without the --recursive flag, the submodules can be fetched by running:
```bash
git submodule update --init --recursive
```

#### profiling
To see the log put:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```