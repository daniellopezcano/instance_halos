# instance_halos

conda create -n VE_IHs python=3.8
conda activate VE_IHs
conda install -c anaconda tensorflow-gpu
conda install jupyter numpy scipy matplotlib networkx cython
git clone instance_halos
pip install -e .

# profiling
To see the log put:

import logging
logging.getLogger().setLevel(logging.DEBUG)