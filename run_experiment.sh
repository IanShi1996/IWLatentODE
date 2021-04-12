#!/bin/bash

#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p t4v2,p100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --job-name=iwlatode
#SBATCH --output=./output/out_%j.txt

source activate diffeq


if [ $2 == "piwae" ]; then
    python ./main_piwae.py --data $1 --model $2 --M $3 --K $4 --ckpt_int $5
    exit 0
fi

if [ "$#" -eq 5 ]; then
    python ./main.py --data $1 --model $2 --M $3 --K $4 --ckpt_int $5
elif [ "$#" -eq 6 ]; then
    python ./main.py --data $1 --model $2 --M $3 --K $4 --beta $5 --ckpt_int $6
fi
