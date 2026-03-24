#!/usr/bin/env bash
#SBATCH --job-name=MusicGen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs-%j.txt # schemat pliku wynikowego

module load anaconda
cd /home/s492459/musical-style-emulation || exit
srun python train.py
