#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-hup-ab
#SBATCH --gpus-per-node=2
#SBATCH --mem=127000M
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

echo "Started GPU!"
nvidia-smi
sleep 3600

# salloc --time=1:00:0 --mem-per-cpu=8G --ntasks=32 --account=def-hup-ab

# Check the status of the machines
# sinfo -o "%20N %8P %5c %8m %10G %6D %9T %E" 

salloc -p gpu --account=def-hup-ab --gres=gpu:l40s:1 --cpus-per-task=32 --mem=256G --time=1:00:00
salloc --account=def-hallett-ab --gpus-per-node=1 --cpus-per-task=16 --mem=128G --time=1:00:00
