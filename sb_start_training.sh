#!/bin/bash

PY_ENV=$1
CONFIG_FILE=$2
BASE_MODEL=$3
RUN_NAME=$4

OUTPUT_DIR="/scratch/local/models/$RUN_NAME/"

#SBATCH --account=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --job-name=nvidia-check
#SBATCH --gres=gpu:3
#SBATCH --ntasks=16
#SBATCH --output=t5_training.%j.out

source "$PY_ENV/bin/activate"

d=$(date '+DATE- %m-%d-%y TIME-%H:%M:%S')
echo "$d"

# identify the latest checkpoint
check_pt=$(ls -t "$OUTPUT_DIR" | head -n 1)
FULL_CHECK_PT_DIR="$OUTPUT_DIR$check_pt"

# If no checkpoint, start from scratch, else continue from chkpt
if [ -z "$check_pt" ]; then 
  echo "No checkpoint. Running from scratch!" ;
  python trainer.py "$CONFIG_FILE" \
                    --model_name_or_path "$FULL_CHECK_PT_DIR" \
                    --output_dir "$OUTPUT_DIR"
                    --run_name "$RUN_NAME"
else 
  echo "Using checkpoint: $full_check_pt_dir";
  python trainer.py "$CONFIG_FILE" \
                    --model_name_or_path "$FULL_CHECK_PT_DIR" \
                    --output_dir "$OUTPUT_DIR" \
                    --run_name "$RUN_NAME" \
                    --check-pt "$FULL_CHECK_PT_DIR"
fi
