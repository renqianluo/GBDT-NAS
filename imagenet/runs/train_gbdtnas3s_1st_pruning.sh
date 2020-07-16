cd ../
export PYTHONPATH=.:$PYTHONPATH

MODEL=GBDT-NAS-3S_1st_Pruning_Net
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/imagenet/raw-data
ARCH="4 4 2 1 6 7 5 3 4 6 5 1 3 2 7 1 6 1 2 6 4"

mkdir -p $OUTPUT_DIR

python train_imagenet.py \
  --data_path=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --lazy_load \
  --arch="$ARCH" \
  --dropout=0.3 \
  --width_stages="32,48,96,104,208,432" \
  | tee -a $OUTPUT_DIR/train.log
