#!/bin/bash
#SBATCH --partition=move --qos=normal --account=move
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=temp 

################################################################################################################################################################################
conda run -n cloc --live-stream python train.py --cfg joint_diffuse.yaml --exp_name fixed_rotation

echo "Done"
