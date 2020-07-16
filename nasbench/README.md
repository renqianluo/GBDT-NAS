# Neural Architecture Search with GBDT

This folder contains the code used for GBDT-NAS on NASBench-101 data set.


## Environments and Requirements
The code is built and tested on Pytorch 1.5

Install nasbench package and download NASBench-101 dataset. 

Install nasbench package install from github (`https://github.com/google-research/nasbench.git`)

Download NASBench-101 dataset from https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
```
mkdir -p data
cd data
wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
cd ..
```

## Searching Architectures
To run the GBDT-NAS, please refer to `runs/train_gbdtnas.sh`:
```
cd runs
bash train_gbdtnas.sh
cd ..
```
After it finishes, it will report discovered architectures and corresponding accuracy.

To evaluate GBDT-NAS for multiple runs and get the average results, please refer to `train_gbdtnas_multi.py`:
```
python train_gbdtnas_multi.py --num_runs=NUM_RUNS # e.g., --num_runs=500
```
It will run for NUM_RUNS times and report the average results at the end of each run.