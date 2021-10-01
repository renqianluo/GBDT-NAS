cd ../
export PYTHONPATH=.:$PYTHONPATH

MODEL=GBDT-NAS-3S_2nd_Pruning_Net
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/imagenet/raw-data
ARCH="5 2 4 6 3 1 1 3 6 5 4 3 6 6 5 4 5 5 4 6"

mkdir -p $OUTPUT_DIR

python train_imagenet.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --lazy_load \
  --arch="$ARCH" \
  --dropout=0.3 \
  --width_stages="32,48,96,104,208,432" \
  | tee -a $OUTPUT_DIR/train.log
