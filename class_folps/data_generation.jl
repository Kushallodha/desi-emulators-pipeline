using Distributed
using SlurmClusterManager
using ArgParse
using Random
using EmulatorsTrainer
using PyCall
using NPZ
using JSON3
using LinearAlgebra



# Argument parsing 
config = ArgParseSettings()
@add_arg_table config begin
    "--emulator_type"
    help = "Specify which emulator type to create: 'effort_folps_pk'."
    arg_type = String
    required = true
    "--parameters"
    help = "Specify (comma separated) the labels for the parameters of the emulator."
    arg_type = String
    required = true
    "--lower_bounds"
    help = "Specify (comma separated) the lower bounds for each of the parameters."
    arg_type = String
    required = true
    "--upper_bounds"
    help = "Specify (comma separated) the upper bounds for each of the parameters."
    arg_type = String
    required = true
    "--n_samples"
    help = "Specify the raw (before filtering) number of data samples to generate."
    arg_type = Int64
    required = true
    "--random_seed"
    help = "Specify (if desired) the random seed so the data samples can be reproduced exactly."
    arg_type = Int64
    default = nothing
    "--root_dir"
    help = "Specify the directory where the data samples will be stored."
    arg_type = String 
    required = true
    "--repo_dir"
    help = "Specify the folder path to desi-emulators-pipeline (depends where it is installed)."
    arg_type = String
    required = true
    "--k_grid_path"
    help = "Specify (if using Effort) the path to read in the desired k grid."
    arg_type = String
    default = nothing
end
parsed_args = parse_args(config)
emulator_type = parsed_args["emulator_type"]
parameters = parsed_args["parameters"]
lower_bounds = parsed_args["lower_bounds"]
upper_bounds = parsed_args["upper_bounds"]
n_samples = parsed_args["n_samples"]
random_seed = parsed_args["random_seed"]
root_dir = parsed_args["root_dir"]
repo_dir = parsed_args["repo_dir"]
k_grid_path = parsed_args["k_grid_path"]
parameters = String.(split(parameters, ","))
lower_bounds = parse.(Float64, split(lower_bounds, ","))
upper_bounds = parse.(Float64, split(upper_bounds, ","))


# Sets up Slurm workers
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["LC_ALL"] = "C.UTF-8"
ENV["LANG"] = "C.UTF-8"

# Sets up Slurm workers
ENV["SLURM_NTASKS"] = ENV["JULIA_TOTAL_TASKS"]
mgr = SlurmManager(;launch_timeout = 600.0, srun_post_exit_sleep = 2.0)
addprocs(mgr)

@everywhere begin
    using Random, EmulatorsTrainer, PyCall, NPZ, JSON3, LinearAlgebra
    sys = pyimport("sys")
    pypath = PyVector(sys["path"])
    if !($repo_dir in pypath)
        pushfirst!(pypath, $repo_dir)
    end
end

@everywhere begin
    random_seed = $random_seed
    n_samples = $n_samples
    parameters = $parameters
    lower_bounds = $lower_bounds
    upper_bounds = $upper_bounds
    emulator_type = $emulator_type
    root_dir = $root_dir
    k_grid_path = $k_grid_path
end

@everywhere begin
    # Generates the latin hypercube samples
    if random_seed != nothing
        Random.seed!(random_seed)
    end
    samples = create_training_dataset(n_samples, lower_bounds, upper_bounds)

    # Filters out bad unphysical samples with w0+wa>0 (if exist)
    if "w0" in parameters && "wa" in parameters
        w0_ind = findfirst(==("w0"), parameters)
        wa_ind = findfirst(==("wa"), parameters)
        w0wa_cond = (samples[w0_ind, :] .+ samples[wa_ind, :]) .< 0
        samples = samples[:, w0wa_cond]
    end

    # Imports necessary python packages from PyCall
    np = pyimport("numpy")
    classy = pyimport("classy")
    folps = pyimport("folps")
    folps_basis = pyimport("folps_basis")
    # Reads in the already computed folps matrices: computed in advance by folps.MatrixCalculator(A_full=true, save_dir=matrix_save_dir).get_mmatrices()
    const matrix = np.load("class_folps/folps_matrices/matrices_nfftlog128_Afull-True_use_TNS-False.npy", allow_pickle=true)[]


    function classy_script(cosmo_dict, root_path)
        """
        Function to generate the necessary data samples for the given set of cosmological parameters.
        
        Arguments:
            'cosmo_dict' -> The dictionary of cosmological parameters associated with the particular data sample.
            'root_path' -> The folder path where the data samples will be stored.
        """ 
        cosmo = nothing
        try
            rand_str = root_path * "/" * randstring(10) # creates folder path for this particular data sample

            # Extracts the cosmological parameters from the dictionary
            h = cosmo_dict["H0"] / 100
            ombh2 = cosmo_dict["ombh2"]
            omch2 = cosmo_dict["omch2"]
            z = haskey(cosmo_dict, "z") ? cosmo_dict["z"] : nothing
            ln10As = haskey(cosmo_dict, "ln10As") ? cosmo_dict["ln10As"] : 3.044 # fixes ln10As, ns to fiducial values for Mapse when only transfer function emulated
            ns = haskey(cosmo_dict, "ns") ? cosmo_dict["ns"] : 0.9649
            w0 = haskey(cosmo_dict, "w0") ? cosmo_dict["w0"] : -1 # fixes w0, wa = -1, 0 if not varied in model
            wa = haskey(cosmo_dict, "wa") ? cosmo_dict["wa"] : 0
            Mnu_in_dict = haskey(cosmo_dict, "Mnu")
            Mnu = Mnu_in_dict ? cosmo_dict["Mnu"] : 0.06 # sum of neutrino mass
            N_ncdm = Mnu_in_dict ? 3 : 1
            m_ncdm = Mnu_in_dict ? "$(Mnu/3),$(Mnu/3),$(Mnu/3)" : Mnu # three denegerate species or single species
            tau = haskey(cosmo_dict, "tau") ? cosmo_dict["tau"] : nothing

            # Sets the cosmological parameters (and adds additional parameters depending on which observable is measured)
            cosmo_params = Dict("ln10^{10}A_s" => ln10As, "n_s" => ns, "h" => h, "omega_b" => ombh2, "omega_cdm" => omch2, 
                                "w0_fld" => w0, "wa_fld" => wa, "use_ppf" => "yes", "fluid_equation_of_state" => "CLP", "cs2_fld" => 1, "Omega_Lambda" => 0, "Omega_scf" => 0,
                                "Neff" => 3.044, "N_ncdm" => N_ncdm, "m_ncdm" => m_ncdm)
            if emulator_type in ["effort_folps_pk"]
                cosmo_params["output"] = "mPk"
                cosmo_params["z_pk"] = string(z)
                cosmo_params["P_k_max_h/Mpc"] = 20
            end

            # Initializes the cosmology
            cosmo = classy.Class()
            cosmo.set(cosmo_params)
            cosmo.compute()

            # Computes and saves statistics
            if emulator_type == "effort_folps_pk"
                konhmin, konhmax, nk = 1e-4, 10, 20000
                konh = np.logspace(np.log10(konhmin), np.log10(konhmax), nk)
                ktarget = npzread(k_grid_path) # desired emulator grid
                plin = [cosmo.pk_cb(k * h, z) * h^3 for k in konh]
                omega_m = float(cosmo.Omega0_m())
                omega_nu = cosmo.Omega_nu
                kwargs = (z=z, h=h, Omega_m=omega_m, fnu=float(omega_nu)/omega_m, f0=float(cosmo.scale_independent_growth_factor_f(z)))
                nonlinear = folps.NonLinearPowerSpectrumCalculator(mmatrices=matrix, kernels="fk"; kwargs...)
                nonlinear.kTout = ktarget 
                nonlinear.nk = length(nonlinear.kTout)
                table, table_now = nonlinear.calculate_loop_table(k=konh, pklin=plin, cosmo=nothing; kwargs...)
                basis = folps_basis.project_basis(table, table_now)
                if any(isnan, basis[0]) || any(isnan, basis[2]) || any(isnan, basis[4])
                    @error "There are nan values!"
                else
                    mkdir(rand_str)
                    npzwrite(rand_str * "/kv.npy", vec(nonlinear.kTout))
                    npzwrite(rand_str * "/pk_0.npy", basis[0])
                    npzwrite(rand_str * "/pk_2.npy", basis[2])
                    npzwrite(rand_str * "/pk_4.npy", basis[4])
                    open(rand_str * "/effort_dict.json", "w") do io 
                        JSON3.write(io, cosmo_dict)
                    end
                end
            end
        catch e
            println("Something went wrong during calculation!")
            println(cosmo_dict)
        finally
            if cosmo != nothing
                try
                    cosmo.struct_cleanup()
                    cosmo.empty()
                catch
                end
            end
        end
    end
end


# Runs code
EmulatorsTrainer.compute_dataset(samples, parameters, root_dir, classy_script, :distributed)