!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=99:00:00
#SBATCH --job-name=fm-random-search-NN
#SBATCH --partition=lopez
#SBATCH --mem=4Gb
#SBATCH --output=%j.o.slurm
#SBATCH --error=%j.e.slurm

export WORKDIR=/scratch/mukadum.f/NN_MolModel/random-eg
export PATH=/home/mukadum.f/miniconda3/bin:$PATH 

cd $WORKDIR

python3 NN-ChemI.py --td data6360.json --iw -2 --ep 1200 --BS 100 500 2  --HL 6 16 2 --ND 30 80 2 --NI 60 --nn eg
