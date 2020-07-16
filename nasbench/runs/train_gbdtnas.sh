cd ..
export PYTHONPATH=.:$PYTHONPATH
MODEL=gbdtnas
OUTPUT_DIR=outputs/$MODEL

mkdir -p $OUTPUT_DIR

python train_gbdtnas.py \
  --output_dir=$OUTPUT_DIR \
  | tee $OUTPUT_DIR/log.txt
