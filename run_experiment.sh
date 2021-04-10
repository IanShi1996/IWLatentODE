#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --job-name=iwlatode
#SBATCH --output=./output/out_%j.txt

source activate diffeq

python ~/Documents/IWLatentODE/main.py

