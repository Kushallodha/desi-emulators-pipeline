# desi-emulators-pipeline

This repository contains the files necessary for training neural network emulators on the NERSC supercomputer.

Important: Many of the settings listed in the current versions of the codes/job scripts are not final/optimized fully (things like the number of training samples, emulator hyperparameters, slurm job settings may need to be adjusted depending on the accuracy/time constraints for the user training the emulator). It is recommend to start out by training an emulator with fewer samples to ensure that everything works and then proceed to larger cases. It is also the user's responsibility to ensure the emulator is free of potential bugs/problems by comparing the emulator outputs with the theory code of interest. (Note that the default neural network architecture has 5 layers of 64 neurons per layer. This may need to be increased/decreased depending on the desired accuracy/runtime tradeoff).

## Installation

First, we must create an environment which permits the use of PyCall within Julia. `cd` to your preferred directory and run the following steps. The second block of instructions is for installing CAMB and CosmoRec which is more involved – please report any problems/bugs to me!

1. 
```
module load python
conda create -n classy_env python=3.10 -y
conda activate classy_env
pip install numpy scipy cython
pip install classy
pip install git+https://github.com/marcobonici/velocileptors_free.git
pip install git+https://github.com/cosmodesi/FolpsD.git
```
```
git clone https://github.com/cmbant/CosmoRec.git
cd CosmoRec

cat << 'EOF' >> Development/Simple_routines/parser.cpp
template int parser_read<int>(const struct file_content&, string, int&, bool&, bool);
template int parser_read<double>(const struct file_content&, string, double&, bool&, bool);
template int parser_read<string>(const struct file_content&, string, string&, bool&, bool);
EOF

sed -i 's/$(CC) $(LXXFLAGS) -L. -lCosmoRec $(LIBS) main.o -o CosmoRec/$(CC) $(LXXFLAGS) main.o -o CosmoRec -L. -lCosmoRec $(LIBS)/' Makefile.in

make clean
make
cd ..

git clone https://github.com/cmbant/CAMB.git
cd CAMB

CR_PATH=$(realpath ../CosmoRec) 

sed -i 's/RECOMBINATION_FILES ?= recfast cosmorec/RECOMBINATION_FILES = recfast cosmorec/' fortran/Makefile_main
sed -i "s|COSMOREC_PATH ?=../../CosmoRec/|COSMOREC_PATH = $CR_PATH|" fortran/Makefile_main
sed -i 's/$(GSoL_LINK)/$(GSL_LINK)/' fortran/Makefile_main

pip install .
cd ..
```

4. Run ```which python``` and copy the folder path (will be needed later).

5. Open a new terminal and clone this repository `git clone https://github.com/jgmorawetz/desi-emulators-pipeline.git`.

6. `cd` to the folder `desi-emulators-pipeline`, and modify the script (can use `vi` for this) `install_julia_environment.jl` by pasting the directory path that was copied from step 2 into the line `ENV["PYTHON"] = `. This step initializes the Julia environment and ensures the necessary dependencies are satisfied (the `Project.toml` and `Manifest.toml` files needed are already located in the folder). Run the following commands:
```
module load julia
julia install_julia_environment.jl
```

## Folder Structure

There are three folders currently `class_velocileptors`, `class_folps` and `camb`. `class_velocileptors` contains all the data generation/training codes and scripts for Effort (when implemented in velocileptors) as well as ACE, Capse and Mapse (since the rest also use class). `class_folps` is for Effort folps specifically, and lastly `camb` is Capse implemented using CAMB as opposed to class.

## Codes and Job Scripts

Each of the above folders contains four main codes and job scripts: `data_generation` and `training` (with .jl and .sh extensions for each). The data generation is responsible for actually generating for samples that will be used for the training, while the training actually optimizes the neural networks.

