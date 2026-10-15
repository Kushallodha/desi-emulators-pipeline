using Pkg
Pkg.activate(".")
using DataFrames
using Effort
using ArgParse
using DelimitedFiles
using Random
using LinearAlgebra
using Statistics
include("training_engine.jl")



# Argument parsing 
config = ArgParseSettings()
@add_arg_table config begin
    "--basis"
    help = "Specify the emulator version 'result_ln10As_basis' or 'result_sigma8_basis'."
    arg_type = String
    required = true
    "--dataset_directory"
    help = "Specify the folder path where the data samples are stored."
    arg_type = String
    required = true
    "--save_directory"
    help = "Specify the directory where the emulator results will be stored (must be created in advance along with subfolders)."
    arg_type = String
    required = true
    "--network_architecture_path"
    help = "Specify the file path where the neural network architecture is specified."
    arg_type = String
    required = true
    "--batchsize"
    help = "Specify the batch size for computing the gradients."
    arg_type = Int 
    required = true
    "--n_epoch"
    help = "Specify the number of epochs per run (for each run results are updated in file)."
    arg_type = Int 
    required = true
    "--n_run"
    help = "Specify the number of runs for each learning rate."
    arg_type = Int 
    required = true 
    "--learning_rates"
    help = "Specify (comma separated) the different learning rates to iterate through."
    arg_type = String
    required = true
end
parsed_args = parse_args(config)
basis = parsed_args["basis"]
dataset_directory = parsed_args["dataset_directory"]
save_directory = parsed_args["save_directory"]
network_architecture_path = parsed_args["network_architecture_path"]
batchsize = parsed_args["batchsize"]
n_epoch = parsed_args["n_epoch"]
n_run = parsed_args["n_run"]
learning_rates = parsed_args["learning_rates"]
learning_rates = parse.(Float64, split(learning_rates, ","))



# Reads the contents of the data samples (and extracts number of samples and input/output dimensions for each)
subdirs = readdir(dataset_directory)
n_samples = length(subdirs)
example_param_dict = JSON.parsefile(joinpath(dataset_directory, subdirs[1], "ace_dict.json"))
n_input = length(example_param_dict) - 1 # since both ln10As and sigma8 are saved but only one is used for input
if haskey(example_param_dict, "w0") && haskey(example_param_dict, "wa") && ! haskey(example_param_dict, "Mnu")
    model = "w0waCDM"
elseif haskey(example_param_dict, "w0") && haskey(example_param_dict, "wa") && haskey(example_param_dict, "Mnu")
    model = "Mnuw0waCDM"
end
n_output = length(npzread(joinpath(dataset_directory, subdirs[1], "$(basis).npy"))) # 7 for the current version of ACE (can change if want more/less quantities)

# Constructs matrices to store the input and output features
input_params = Matrix{Float64}(undef, n_samples, n_input)
targets = Matrix{Float64}(undef, n_samples, n_output)


# Goes through the data samples and adds input/target features and then at end constructs dataframe
for (idx, sub) in enumerate(subdirs)
    if idx % 50000 == 0
        println("Loaded up to $(idx) samples.")
    end
    sub_path = joinpath(dataset_directory, sub)
    param_dict = JSON.parsefile(joinpath(sub_path, "ace_dict.json"))
    # Determines whether w0waCDM or Mnuw0waCDM model
    if model == "w0waCDM"
        if basis == "result_ln10As_basis"
            z, ln10As, ns, H0, ombh2, omch2, w0, wa = (param_dict[param] for param in ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "w0", "wa"])
            Mnu = 0.06
            input_params[idx, :] = [z, ln10As, ns, H0, ombh2, omch2, w0, wa]
        elseif basis == "result_sigma8_basis"
            z, sigma8, ns, H0, ombh2, omch2, w0, wa = (param_dict[param] for param in ["z", "sigma8", "ns", "H0", "ombh2", "omch2", "w0", "wa"])
            Mnu = 0.06
            input_params[idx, :] = [z, sigma8, ns, H0, ombh2, omch2, w0, wa]
        end
    elseif model == "Mnuw0waCDM"
        if basis == "result_ln10As_basis"
            z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa = (param_dict[param] for param in ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"])
            input_params[idx, :] = [z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa]
        elseif basis == "result_sigma8_basis"
            z, sigma8, ns, H0, ombh2, omch2, Mnu, w0, wa = (param_dict[param] for param in ["z", "sigma8", "ns", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"])
            input_params[idx, :] = [z, sigma8, ns, H0, ombh2, omch2, Mnu, w0, wa]   
        end         
    end
    target_file = npzread(joinpath(sub_path, "$(basis).npy"))
    targets[idx, :] = vec(target_file) # no preprocessing for ACE
end
if model == "w0waCDM"
    if basis == "result_ln10As_basis"
        df = DataFrame(z=input_params[:, 1], ln10As=input_params[:, 2], ns=input_params[:, 3], 
                       H0=input_params[:, 4], ombh2=input_params[:, 5], omch2=input_params[:, 6],
                       w0=input_params[:, 7], wa=input_params[:, 8],
                       observable=[targets[i, :] for i in 1:n_samples])
    elseif basis == "result_sigma8_basis"
        df = DataFrame(z=input_params[:, 1], sigma8=input_params[:, 2], ns=input_params[:, 3], 
                       H0=input_params[:, 4], ombh2=input_params[:, 5], omch2=input_params[:, 6],
                       w0=input_params[:, 7], wa=input_params[:, 8],
                       observable=[targets[i, :] for i in 1:n_samples])
    end
elseif model == "Mnuw0waCDM"
    if basis == "result_ln10As_basis"
        df = DataFrame(z=input_params[:, 1], ln10As=input_params[:, 2], ns=input_params[:, 3], 
                       H0=input_params[:, 4], ombh2=input_params[:, 5], omch2=input_params[:, 6],
                       Mnu=input_params[:, 7], w0=input_params[:, 8], wa=input_params[:, 9],
                       observable=[targets[i, :] for i in 1:n_samples])
    elseif basis == "result_sigma8_basis"
        df = DataFrame(z=input_params[:, 1], sigma8=input_params[:, 2], ns=input_params[:, 3], 
                       H0=input_params[:, 4], ombh2=input_params[:, 5], omch2=input_params[:, 6],
                       Mnu=input_params[:, 7], w0=input_params[:, 8], wa=input_params[:, 9],
                       observable=[targets[i, :] for i in 1:n_samples])
    end
end

# Now runs the training itself
run_training(df, save_directory, network_architecture_path, learning_rates, n_run, n_epoch, batchsize)