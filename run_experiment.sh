#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --job-name=iwlatode
#SBATCH --output=./output/out_%j.txt

source activate diffeq

if [ "$#" -eq 4 ]; then
    python ~/Documents/IWLatentODE/main.py --model $1 --M $2 --K $3 --ckpt_int $4
elif [ "$#" -eq 5 ]; then
    python ~/Documents/IWLatentODE/main.py --model $1 --M $2 --K $3 --beta $4 --ckpt_int $5
fi
