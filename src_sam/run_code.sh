#!/bin/bash
#SBATCH --job-name=test-101
#SBATCH --account=yuni0
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=64g
#SBATCH --mail-user=zhonglil@med.umich.edu
#SBATCH --mail-type=ALL
#SBATCH --output=./%x-%j
if [[ $SLURM_JOB_NODELIST ]] ; then
echo "Running on"
scontrol show hostnames $SLURM_JOB_NODELIST
fi
module load python3.9-anaconda/2021.11
export PYTHONPATH=$PWD:$PYTHONPATH
python3 train_test_zhonglin.py
