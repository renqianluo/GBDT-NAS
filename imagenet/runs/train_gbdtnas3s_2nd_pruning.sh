cd ../
export PYTHONPATH=.:$PYTHONPATH

MODEL=GBDT-NAS-3S_2nd_Pruning_Net
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/imagenet/raw-data
ARCH="5 1 6 7 6 7 1 7 4 7 7 4 6 3 6 7 2 1 5 4 6"

mkdir -p $OUTPUT_DIR

python train_imagenet.py \
  --data_path=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --lazy_load \
  --arch="$ARCH" \
  --dropout=0.3 \
  --width_stages="32,56,112,128,256,432" \
  | tee -a $OUTPUT_DIR/train.log
