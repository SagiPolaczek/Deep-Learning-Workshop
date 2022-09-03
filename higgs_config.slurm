#! /bin/sh

#SBATCH --job-name=higgs_config
#SBATCH --output=higgs_experiments/higgs_config.out # redirect stdout
#SBATCH --error=higgs_experiments/higgs_config.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=2000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=10 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python awesome.py