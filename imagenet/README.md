# Neural Architecture Search with GBDT

This folder contains the code used for GBDT-NAS and GBDT-NAS-3S on ImageNet data set.


## Environments and Requirements
The code is built and tested on Pytorch 1.5

Require: [`lightgbm`](https://github.com/microsoft/LightGBM), [`SHAP`](https://github.com/slundberg/shap)

You can install them via pip:
```
pip install lightgbm shap
```

## Searching Architectures
To run the search process, please refer to `runs/search_gbdtnas.sh`, `runs/search_gbdtnas3s_1st_pruning.sh` and `runs/search_gbdtnas3s_2nd_pruning.sh` for GBDT-NAS, GBDT-NAS-3S (first-order pruning) and GBDT-NAS-3S (second-order pruning) respectively.
Point `DATA_DIR` in the scripts to the path of imagenet dataset on your device. It requires 4 GPU cards to train.
```
cd runs
bash search_gbdtnas.sh
bash search_gbdtnas3s_1st_pruning.sh
bash search_gbdtnas3s_2nd_pruning.sh
```
After the search finishes, you can find the final discovored top architectures in the $OUTPUT_DIR defined in the script. The final discovered top architectures are listed in the file `arch_pool.3` (since the algorithm runs for 3 iterations) where each line is an architecture represented in sequence. Each token in the sequence represents a specific operation. The mapping of the token and correspoding operation is shown in `utils.py`.

## Training Architectures

### Training the architecture discovered in our paper
To directly train the architecture discovered as we report in the paper, please refer to `runs/train_gbdtnas.sh`, `runs/train_gbdtnas3s_1st_pruning.sh` and `runs/train_gbdtnas3s_2nd_pruning.sh` for architecture discovered by GBDT-NAS, GBDT-NAS-3S (first-order pruning) and GBDT-NAS-3S (second-order purning) respectively. It requires 4 GPU cards to train.
```
cd runs
bash train_gbdtnas.sh
bash train_gbdtnas3s_1st_pruning.sh
bash train_gbdtnas3s_2nd_pruning.sh
```

### Training customized architectures
To train a customized architecture (e.g., discovered by GBDT-NAS or GBDT-NAS-3S of your own run), you can modify the script `runs/train_gbdtnas.sh` by replacing the `ARCH` with your customized net architecture sequence string (e.g., top architecture in the `arch_pool.3` file).

## Evaluating the architecture discovered in our paper
We provide the checkpoint of the architectures in [`Google Dirve`](https://drive.google.com/drive/folders/1pzpeCiOp3AyBTkXkBlBIbui8dPR1iNoo?usp=sharing)
You can download the checkpoints and unzip them to the current path:
```
unzip GBDT-NAS_Net.zip
unzip GBDT-NAS-3S_1st_Pruning_Net.zip
unzip GBDT-NAS-3S_2nd_Pruning_Net.zip
```
To evalute them, please refer to `runs/evaluate.sh`.