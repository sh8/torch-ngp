#!/bin/bash
#YBATCH -r dgx-a100_1
#SBATCH -N 1
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH -J thirdeye

. /etc/profile.d/modules.sh
module load cuda/11.5
module load cudnn

python main_nerf_gan.py data/ShapeNetCore.v2 \
  --workspace trial_nerf_shapenet \
  --fp16 --tcnn --mode shapenet \
  --bound 1.0 --scale 3.0 --dt_gamma 0
