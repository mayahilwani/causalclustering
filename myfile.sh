#!/bin/bash

#SBATCH --job-name=mayaj
#SBATCH --output=/tmp/job-%j.out
#SBATCH --partition=r65257773x
#SBATCH --cpus-per-task=32

JOBDATADIR=`ws create work --space $SLURM_JOB_ID --duration "1 00:00:00"`
JOBTMPDIR=/tmp/job-$SLURM_JOB_ID

# test for the credentials files
srun test -f ~/CISPA-home/.config/enroot/.credentials

srun mkdir $JOBTMPDIR

cd ~/CISPA-home/causalclustering
export PYTHONPATH=$(pwd)
export OMP_NUM_THREADS="1"

srun --container-image=projects.cispa.saarland:5005#c01mahi/mydockermain:v1 --container-mounts=$JOBTMPDIR:/tmp --pty bash -c "source /opt/conda/bin/activate && conda activate myenv && cd $HOME/CISPA-home/causalclustering && python3 $HOME/CISPA-home/causalclustering/script.py $1 $2 $3 $4 $5 $6 $7 $8 $9"

srun mv /tmp/job-"$SLURM_JOB_ID".out "$JOBDATADIR"/out.txt
srun mv "$JOBTMPDIR" "$JOBDATADIR"/data

