#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training_mapse_class
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "cb"
    "mm"
)

PARAM_SET=(${PARAMS[$SLURM_ARRAY_TASK_ID]})
spectrum="${PARAM_SET}"

home_dir="/global/homes/j/jgmorawe/desi-emulators-pipeline"
scratch_dir="/pscratch/sd/j/jgmorawe"
dataset_directory="${scratch_dir}/mapse_class_mnuw0wacdm_100000"
save_directory="${home_dir}/trained_mapse_class_mnuw0wacdm_100000/Pk_lin_${spectrum}"
network_architecture_path="${home_dir}/nn_setup_64x5.json"
var_ratio=0.999999
n_epoch=2000
n_run=30
batchsize=1024
learning_rates="1e-4,3e-5,1e-5,3e-6,1e-6,3e-7"

julia -t $SLURM_CPUS_PER_TASK class_velocileptors/training_mapse.jl --spectrum="$spectrum" --dataset_directory="$dataset_directory" --save_directory="$save_directory" --var_ratio=$var_ratio --network_architecture_path="$network_architecture_path" --batchsize=$batchsize --n_epoch=$n_epoch --n_run=$n_run --learning_rates="$learning_rates"