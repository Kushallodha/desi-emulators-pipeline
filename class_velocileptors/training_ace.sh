#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training_ace_class
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "ln10As_basis"
    "sigma8_basis"
)

PARAM_SET="${PARAMS[$SLURM_ARRAY_TASK_ID]}"
basis="result_${PARAM_SET}"

home_dir="/global/homes/j/jgmorawe/desi-emulators-pipeline"
scratch_dir="/pscratch/sd/j/jgmorawe"
dataset_directory="${scratch_dir}/ace_class_mnuw0wacdm_100000"
save_directory="${home_dir}/trained_ace_class_${PARAM_SET}_mnuw0wacdm_100000"
save_directory_before="${home_dir}/trained_ace_class_${PARAM_SET}_mnuw0wacdm_100000"
network_architecture_path="${home_dir}/nn_setup_64x5.json"
n_epoch=2000
n_run=30
batchsize=1024
learning_rates="1e-4,3e-5,1e-5,3e-6,1e-6,3e-7"

cp "${home_dir}/class_velocileptors/supporting_files/ACE/postprocessing.py"      "${save_directory}/postprocessing.py"
cp "${home_dir}/class_velocileptors/supporting_files/ACE/postprocessing.jl"      "${save_directory}/postprocessing.jl"


julia -t $SLURM_CPUS_PER_TASK class_velocileptors/training_ace.jl --basis="$basis" --dataset_directory="$dataset_directory" --save_directory="$save_directory" --network_architecture_path="$network_architecture_path" --batchsize=$batchsize --n_epoch=$n_epoch --n_run=$n_run --learning_rates="$learning_rates"