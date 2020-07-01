# MetaIMC
- This is the code for our paper *Meta-Learning Bayes Priors For Inductive Matrix Completion
***********************************************************

## Datasets
* Flixster
* Douban
* YahooMusic

***************************************************************

## Requirements
* Ubuntu 16.0.4
* Python 3.8
* Pytorch 1.4
* PyTorch_Geometric 1.4
Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.
***************************************************************
***************************************************************

## Preprocessing
```
python3 dataProcess.py 
```
***************************************************************

## Training
```
python Main.py --data-name flixster --epochs 40 --testing --ensemble
```
The results will be saved in "results/flixster_testmode/". The processed enclosing subgraphs will be saved in "data/flixster/testmode/". Change flixster to douban or yahoo_music to do the same experiments on Douban and YahooMusic datasets, respectively. Delete --testing to evaluate on a validation set to do hyperparameter tuning.

