cd ../
export PYTHONPATH=.:$PYTHONPATH

MODEL=SEARCH_GBDTNAS
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/imagenet/raw-data

mkdir -p $OUTPUT_DIR

python search_gbdtnas.py \
  --data_path=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --lazy_load \
  | tee -a $OUTPUT_DIR/train.log
