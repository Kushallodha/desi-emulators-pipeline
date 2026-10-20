#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training_effort_velocileptors
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "11 0"
    "11 2"
    "11 4"
    "loop 0"
    "loop 2" 
    "loop 4"
    "ct 0"
    "ct 2"
    "ct 4"
)

PARAM_SET=(${PARAMS[$SLURM_ARRAY_TASK_ID]})
component="${PARAM_SET[0]}"
multipole="${PARAM_SET[1]}"

home_dir="/global/homes/j/jgmorawe/desi-emulators-pipeline"
scratch_dir="/pscratch/sd/j/jgmorawe"
dataset_directory="${scratch_dir}/effort_velocileptors_rept_mnuw0wacdm_100000"
save_directory="${home_dir}/trained_effort_velocileptors_rept_mnuw0wacdm_100000/${multipole}/${component}"
save_directory_before="${home_dir}/trained_effort_velocileptors_rept_mnuw0wacdm_100000/${multipole}"
network_architecture_path="${home_dir}/nn_setup_64x5.json"
n_epoch=2000
n_run=30
batchsize=1024
learning_rates="1e-4,3e-5,1e-5,3e-6,1e-6,3e-7"


cp "${home_dir}/class_velocileptors/supporting_files/biascombination.py"    "${save_directory_before}/biascombination.py"
cp "${home_dir}/class_velocileptors/supporting_files/biascombination.jl"    "${save_directory_before}/biascombination.jl"
cp "${home_dir}/class_velocileptors/supporting_files/jacbiascombination.py" "${save_directory_before}/jacbiascombination.py"
cp "${home_dir}/class_velocileptors/supporting_files/jacbiascombination.jl" "${save_directory_before}/jacbiascombination.jl"

cp "${home_dir}/class_velocileptors/supporting_files/stochmodel_${multipole}.py" "${save_directory_before}/stochmodel.py"
cp "${home_dir}/class_velocileptors/supporting_files/stochmodel_${multipole}.jl" "${save_directory_before}/stochmodel.jl"

if [ "${component}" = "loop" ]; then
    cp "${home_dir}/class_velocileptors/supporting_files/postprocessing_loop.py" "${save_directory}/postprocessing.py"
    cp "${home_dir}/class_velocileptors/supporting_files/postprocessing_loop.jl" "${save_directory}/postprocessing.jl"
else
    cp "${home_dir}/class_velocileptors/supporting_files/postprocessing.py"      "${save_directory}/postprocessing.py"
    cp "${home_dir}/class_velocileptors/supporting_files/postprocessing.jl"      "${save_directory}/postprocessing.jl"
fi

cp "${home_dir}/class_velocileptors/supporting_files/stochmodel_${multipole}.py" "${save_directory}/stochmodel.py"
cp "${home_dir}/class_velocileptors/supporting_files/stochmodel_${multipole}.jl" "${save_directory}/stochmodel.jl"

cp "${home_dir}/k.npy" "${save_directory}/k.npy"


julia -t $SLURM_CPUS_PER_TASK class_velocileptors/training_effort.jl --multipole="$multipole" --component="$component" --dataset_directory="$dataset_directory" --save_directory="$save_directory" --network_architecture_path="$network_architecture_path" --batchsize=$batchsize --n_epoch=$n_epoch --n_run=$n_run --learning_rates="$learning_rates"