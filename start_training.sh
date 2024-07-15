#! /bin/bash

workspace=$1
repo="$workspace/t5_trainer"
py_env="$workspace/my_py"



# Check if the directory /workspace/t5_trainer exists
if [ ! -d "$repo" ]; then
  echo "Cloning code"
  # If the directory does not exist, clone the repository and install the requirements
  git clone https://github.com/ahmeshaf/t5_trainer.git "$repo"
fi

# Check if the environment exits
if [ ! -d "$py_env" ]; then
  echo "Installing libs"
  # if not, create a new environment and install requirements
  python3 -m venv "$py_env"
  source "$py_env/bin/activate"
  pip install -r --no-cache-dir "$repo/requirements.txt"
fi

source "$py_env/bin/activate"

d=$(date '+DATE- %m-%d-%y TIME-%H:%M:%S')
echo "$d"

# identify the latest checkpoint
output_dir=$(jq -r '.trainer.output_dir' "$repo/config.json") && check_pt=$(ls -t "$output_dir" | head -n 1)
full_check_pt_dir="$output_dir$check_pt"

echo "using checkpoint: $full_check_pt_dir"

python trainer.py events-synergy/entsum_processed \
                  --config-file "$repo/config.json" \
                  --check-pt "$full_check_pt_dir" \
                  --kv "model_name_or_path=$full_check_pt_dir"
