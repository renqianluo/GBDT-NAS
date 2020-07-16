cd ../
export PYTHONPATH=.:$PYTHONPATH

MODEL=GBDT-NAS_Net
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/imagenet/raw-data
ARCH="6 6 5 5 2 5 1 4 2 7 4 2 4 5 1 7 6 4 2 4 6"

mkdir -p $OUTPUT_DIR

python train_imagenet.py \
  --data_path=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --lazy_load \
  --arch="$ARCH" \
  --dropout=0.3 \
  --width_stages="32,40,80,104,208,432" \
  | tee -a $OUTPUT_DIR/train.log
