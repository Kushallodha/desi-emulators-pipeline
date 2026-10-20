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

There are three folders currently `class_velocileptors`, `class_folps` and `camb`. `class_velocileptors` contains all the data generation/training codes and scripts for Effort (when implemented in velocileptors) as well as ACE, Capse and Mapse (since the rest also use class). `class_folps` is for Effort folps specifically, and lastly `camb` is Capse implemented using CAMB as opposed to class. Each of these folders also contain the various test notebooks and supporting files required for each emulator version.

## Codes and Job Scripts

Each of the above folders contains four main codes and job scripts: `data_generation` and `training` (with .jl and .sh extensions for each). The data generation is responsible for actually generating for samples that will be used for the training, while the training actually optimizes the neural networks. We briefly review the necessary settings for each:

1. `data_generation.jl` – This code creates the data samples used to train the emulators. Comments are provided throughout explaining it.

2. `data_generation.sh` – This script runs `data_generation.jl`. The user must specify: the emulator type being trained, the parameters (can include any of the following 'ln10As', 'ns', 'H0', 'ombh2', 'omch2', 'tau', 'Mnu', 'w0', 'wa' and only these, unless the user generalizes the code), the lower and upper bounds for each parameter, the total number of samples to include (note: this is before the cutoff w0+wa<0 is applied so in reality there will be fewer than this), the random seed for the data samples so results are reproducible (or can set to nothing if don't want to specify), the directory to store away the data samples (cannot already exist or will cause an error message – also must run `rm .dataset_metadata.json` after the data generation is finished before running the training), and for Effort specifically the desired k grid for the emulator. The code is currently tailored for Mnuw0waCDM model but the user can change this if certain parameters are to be fixed or new ones introduced. The .sh settings will also need to be changed by the user depending on the computational intensity of the desired emulator.

3. `training.jl` – This code takes in the data samples and does the training of the neural networks. Each of the versions across the different emulators are different in how they read in/preprocess the data samples, but they all then proceed to run the training steps found in `training_engine.jl` where the network is initialized based on the architecture setup found in `nn_setup_64x5.json` (currently set up to have 64 neurons per layer and 5 layers, but the user can change these if they want depending on the accuracy/runtime tradeoff). For Effort, the preprocessing involves rescaling the components by the primordial amplitude and growth factor squared (or this combination squared for the loop component) to effectively remove parts of the calculation that can be done analytically and postprocess them back in at the end. A similar thing is done for Capse but also involving the optical depth tau. For ACE these is no pre/post processing. And for Mapse, the primordial power spectrum (involving As and ns) is removed entirely from the emulation process so they are not even parameters. Also for Mapse, an SVD compression is done to reduce the number of elements needed for training. And lastly, for the Capse emulators (since they now go up to higher l), the data samples/training are done on a coarser lobatto grid and then cubic spline interpolation is used to output onto the full l grid to avoid the computational cost of emulating the entire l grid. During the training process itself, inputs and outputs are normalized between their min/max values and then once the emulator is run they are un-normalized. To ensure this doesn't cause any problems, it is important to check in advance that the min/max range of the outputs are sensible (sometimes code instabilities can cause spurious samples that are many orders of magnitude higher than the rest of the samples, so these need to be removed in advance to stop the training from failing). Furthermore, as the training occurs, weights are continuously updated but only saved away to file if the test loss continues to decrease, this prevents overfitting in case the test loss starts to increase again.

4. `training.sh` – This script runs `training.jl`. The user must specify: the components being trained (in the case of Effort this is the multipole 0/2/4 and the component 11/loop/ct, for the Capse it is the spectra TT/TE/EE/PP, for ACE it is the basis ln10As/sigma8 and for Mapse it is cb/mm power spectrum). Note that the training folders that you specify must be created in advance of the running or else it will throw a bug (this included all the subfolders for each), the directory where the training samples are stored, the directory for saving the training results (mentioned above), as well as emulator hyperparameters such as the path to the NN setup file, the batchsize, the number of runs (the number times to run through a given number of epochs, and results are updated at end of each run), the number of epochs per run, and a list of the learning rates to iterate through. Again, the parameters listed in the existing scripts are not necessarily optimized and the user will likely need to updated on case by case basis.

Note that all of these must be run from the emulator home directory in order to import packages/scripts correctly. Furthermore, there are some absolute folder paths in the job scripts that will need to just adjusted for each user.

## Notebooks

There are also notebooks attached showing the basics of how to load emulators after training them and how to compare with the exact theory node (note: these versions are currently out of date and some aspects may need to be adjusted for different emulator versions).

## Supporting Files

As suggested before, each of the codes have additional files found in the `supporting_files` subfolders. All of the codes have `postprocessing.jl` files which tell the emulator how to postprocess the output (basically just undoing the rescaling that was done prior to training). Note that these postprocessing.jl files assume specific indices for the input parameters (e.g. Capse assumes ln10As is the second parameter and tau is the sixth parameter), if the user wishes to change the input parameters they will also need to change these manually! For Effort specifically, there are also `biascombination.jl` and `jacbiascombination.jl` files which tell the emulator how to take linear combinations of the various emulator components in terms of the EFT parameters to calculate the final power spectrum (and the associated jacobian for derivatives). And there are also `stochmodel.jl` files which encode how the stochastic terms combine with the 11/loop/ct terms when doing the final summation, note that these contributions are fully analytic and don't rely on emulator components.
