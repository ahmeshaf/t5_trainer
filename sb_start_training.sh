#!/bin/sh

#SBATCH --account=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --job-name=t5_training
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --output=t5_training.%j.out

# Load modules
echo "Loading modules..."
module load python

export TOKENIZERS_PARALLELISM=false

# echo HF_HOME
echo "HF_HOME: $HF_HOME"

# Activate conda environment
echo "Activating conda environment at $SCRATCHDIR/$USER/venv/"
source activate "$SCRATCHDIR/$USER/venv/bin/activate"

OUTPUT_DIR="$SCRATCHDIR/$USER/models/srl-xl-conll05"

# Start training
echo "Starting training..."
python trainer.py configs/srl_conll05.json \
                  --output-dir "$OUTPUT_DIR" \
                  --continue-training