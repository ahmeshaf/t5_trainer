#! /bin/bash
export HF_HOME=/root/.cache/huggingface/

apt install -y jq;

CONFIG_FILE=/workspace/t5_trainer/configs/srl_ds_elaboration.json
PYTHON_FILE=/workspace/t5_trainer/trainer.py

source /workspace/venv/bin/activate

# identify the latest checkpoint
OUTPUT_DIR=$(jq -r '.trainer.output_dir' "$CONFIG_FILE") && check_pt=$(ls -t "$OUTPUT_DIR" | head -n 1)
full_check_pt_dir="$OUTPUT_DIR$check_pt"

# If no checkpoint, start from scratch, else continue from chkpt
if [ -z "$check_pt" ]; then 
  echo "No checkpoint. Running from scratch!" && python "$PYTHON_FILE" "$CONFIG_FILE"
else 
  echo "Using checkpoint: $full_check_pt_dir" && python "$PYTHON_FILE" "$CONFIG_FILE" --check-pt "$full_check_pt_dir"
fi
