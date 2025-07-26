#!/bin/bash
#SBATCH --job-name=clip_train
#SBATCH --account=gts-mroozbahani3
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o clip_train_%j.out
#SBATCH -e clip_train_%j.err
#SBATCH --mail-user=spatel776@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL


cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate clip-env

# cd src

python train_models.py --arch clip_vitb32 \
  --name final_clip_run \
  --train ../data/labels/label_final_train.tsv \
  --dev ../data/labels/label_final_val.tsv \
  --test ../data/labels/label_final_test.tsv \
  --num-class 4 \
  --batch-size 32 \
  --num-epochs 10 \
  --learning-rate 1e-4 \
  --keep-frozen True \
  --data-dir ../data/images/ 
