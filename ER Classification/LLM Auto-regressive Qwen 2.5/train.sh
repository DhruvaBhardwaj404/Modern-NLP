# This script should run the training and save checkpoints in the output/ dir
OUTPUT_DIR=$1

python convert_data.py "$OUTPUT_DIR"
python main_q2.py train "$OUTPUT_DIR"