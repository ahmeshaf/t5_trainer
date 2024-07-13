#! /bin/bash

# Check if the directory /workspace/t5_trainer exists
if [ ! -d "/workspace/t5_trainer" ]; then
  echo "Cloning code"
  # If the directory does not exist, clone the repository and install requirements
  git clone https://github.com/ahmeshaf/t5_trainer.git /workspace/t5_trainer
fi

# Check if the environment exits
if [ ! -d "/workspace/my_py" ]; then
  echo "Installing libs"
  # if not, create a new environment and install requirements
  python3 -m venv /workspace/my_py
  source /workspace/my_py/bin/activate
  pip install -r /workspace/t5_trainer/requirements.txt
fi

source /workspace/my_py/bin/activate

cd /workspace/t5_trainer/
d=$(date '+DATE- %m-%d-%y TIME-%H:%M:%S')
echo "$d"

output_dir=$(jq -r '.trainer.output_dir' ./config.json) && check_pt=$(ls -t "$output_dir" | head -n 1)
full_check_pt_dir="$output_dir$check_pt"

echo "using checkpoint: $full_check_pt_dir"

python trainer.py events-synergy/entsum_processed --check-pt "$full_check_pt_dir" --kv "model_name_or_path=$full_check_pt_dir" > "continue_training.txt" &
