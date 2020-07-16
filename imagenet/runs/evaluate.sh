cd ../
export PYTHONPATH=.:$PYTHONPATH

#CHECKPOINT=GBDT-NAS_Net/checkpoint.pt
#CHECKPOINT=GBDT-NAS-3S_1st_Pruning_Net/checkpoint.pt
CHECKPOINT=GBDT-NAS-3S_2nd_Pruning_Net/checkpoint.pt
DATA_DIR=data/imagenet/raw-data

python eval.py \
  --data_path=$DATA_DIR \
  --path=$CHECKPOINT
