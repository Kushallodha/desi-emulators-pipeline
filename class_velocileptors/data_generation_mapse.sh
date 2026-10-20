#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name=datageneration_mapse_class
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=300G


module load julia

export JULIA_DEPOT_PATH=/global/homes/j/jgmorawe/.julia
export JULIA_PROJECT=/global/homes/j/jgmorawe/desi-emulators-pipeline
export JULIA_WORKER_TIMEOUT=600
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

julia --project=$JULIA_PROJECT -e 'using Pkg; Pkg.instantiate();'

export JULIA_TOTAL_TASKS=$(
  scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | grep -m1 '^NumTasks=' | cut -d= -f2
)

emulator_type="mapse"
parameters="z,H0,ombh2,omch2,Mnu,w0,wa"
lower_bounds="0.2,50,0.02,0.08,0,-3,-4"
upper_bounds="1.6,85,0.024,0.16,0.5,1,3"
n_samples=100000
random_seed=0
root_dir="/pscratch/sd/j/jgmorawe/mapse_class_mnuw0wacdm_100000"

srun -N1 -n1 --mpi=none --cpu-bind=none --overlap julia --project=$JULIA_PROJECT class_velocileptors/data_generation.jl --emulator_type="$emulator_type" --parameters="$parameters" --lower_bounds="$lower_bounds" --upper_bounds="$upper_bounds" --n_samples=$n_samples --random_seed=$random_seed --root_dir="$root_dir"