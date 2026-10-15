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
    help = "Specify which emulator type to create: 'capse'."
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
end
parsed_args = parse_args(config)
emulator_type = parsed_args["emulator_type"]
parameters = parsed_args["parameters"]
lower_bounds = parsed_args["lower_bounds"]
upper_bounds = parsed_args["upper_bounds"]
n_samples = parsed_args["n_samples"]
random_seed = parsed_args["random_seed"]
root_dir = parsed_args["root_dir"]
parameters = String.(split(parameters, ","))
lower_bounds = parse.(Float64, split(lower_bounds, ","))
upper_bounds = parse.(Float64, split(upper_bounds, ","))



# Sets up Slurm workers
ENV["SLURM_NTASKS"] = ENV["JULIA_TOTAL_TASKS"]
mgr = SlurmManager(;launch_timeout = 600.0, srun_post_exit_sleep = 2.0)
addprocs(mgr)

@everywhere begin
    using Random, EmulatorsTrainer, PyCall, NPZ, JSON3, LinearAlgebra
end

@everywhere begin
    random_seed = $random_seed
    n_samples = $n_samples
    parameters = $parameters
    lower_bounds = $lower_bounds
    upper_bounds = $upper_bounds
    emulator_type = $emulator_type
    root_dir = $root_dir
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
    camb = pyimport("camb")


    function camb_script(cosmo_dict, root_path)
        """
        Function to generate the necessary data samples for the given set of cosmological parameters.
        
        Arguments:
            'cosmo_dict' -> The dictionary of cosmological parameters associated with the particular data sample.
            'root_path' -> The folder path where the data samples will be stored.
        """ 
        pars = nothing
        results = nothing
        cmb = nothing
        lens = nothing
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
            tau = haskey(cosmo_dict, "tau") ? cosmo_dict["tau"] : nothing

            # Sets the cosmological parameters (and adds additional parameters depending on which observable is measured)
            lmax = 10000
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=h*100, ombh2=ombh2, omch2=omch2, omk=0, neutrino_hierarchy="degenerate", 
                               num_massive_neutrinos=N_ncdm, mnu=Mnu, nnu=3.044, TCMB=2.7255)
            pars.InitPower.set_params(As=np.exp(ln10As)*1e-10, ns=ns, pivot_scalar=0.05)
            pars.Reion.set_tau(tau)
            pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
            pars.NonLinear = camb.model.NonLinear_both
            pars.NonLinearModel.set_params(halofit_version="mead2020")
            pars = camb.set_params(cp=pars, recombination_model="CosmoRec", kmax=10, k_per_logint=130, lens_potential_accuracy=8, lens_output_margin=2050,
                                   AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1.2, min_l_logl_sampling=6000, DoLateRadTruncation=false,
                                   halofit_version="mead2020", lmax=lmax)
            results = camb.get_results(pars)


            # Computes and saves statistics
            if emulator_type == "capse"
                # Computes the temperature, polarization, lensing angular CMB power spectra quantities
                cmb = results.get_lensed_scalar_cls(CMB_unit="muK", raw_cl=true)
                lens = results.get_lens_potential_cls(lmax=lmax, raw_cl=true)
                ell_full = np.arange(lmax+1)
                mask = ell_full .>= 2
                ell = ell_full[mask]
                dl = ell_full .* (ell_full .+ 1) ./ (2 * np.pi)
                pp_factor = (ell_full .* (ell_full .+ 1)) .^ 2 ./ (2 * np.pi)
                tt = (cmb[1:lmax+1, 1] .* dl)[mask]
                te = (cmb[1:lmax+1, 4] .* dl)[mask]
                ee = (cmb[1:lmax+1, 2] .* dl)[mask]
                pp = (lens[1:lmax+1, 1] .* pp_factor)[mask]
                if any(isnan, tt) || any(isnan, te) || any(isnan, ee) || any(isnan, pp)
                    @error "There are nan values!"
                else
                    mkdir(rand_str)
                    npzwrite(rand_str * "/TT.npy", tt)
                    npzwrite(rand_str * "/TE.npy", te)
                    npzwrite(rand_str * "/EE.npy", ee)
                    npzwrite(rand_str * "/PP.npy", pp)
                    open(rand_str * "/capse_dict.json", "w") do io
                        JSON3.write(io, cosmo_dict)
                    end
                end
            end
        catch e
            println("Something went wrong during calculation!")
            println(cosmo_dict)
        finally 
            pars = nothing
            results = nothing
            cmb = nothing
            lens = nothing
            GC.gc()
            pyimport("gc").collect()
        end
    end
end


# Runs code
EmulatorsTrainer.compute_dataset(samples, parameters, root_dir, camb_script, :distributed)