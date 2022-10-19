using Pumas
using PumasUtilities
using PharmaDatasets

# Read in Data
pkfile = dataset("iv_sd_3")

# Convert DataFrame into Collection of Subjects (Population)
population = read_pumas(pkfile)

# Model definition
model = @model begin

    @param begin
        # here we define the parameters of the model
        tvcl ∈ RealDomain(; lower = 0.001) # typical clearance 
        tvvc ∈ RealDomain(; lower = 0.001) # typical central volume of distribution
        Ω ∈ PDiagDomain(2)             # between-subject variability
        σ ∈ RealDomain(; lower = 0.001)    # residual error
    end

    @random begin
        # here we define random effects
        η ~ MvNormal(Ω) # multi-variate Normal with mean 0 and covariance matrix Ω
    end

    @pre begin
        # pre computations and other statistical transformations
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
    end

    # here we define compartments and dynamics
    @dynamics Central1 # same as Central' = -(CL/Vc)*Central (see Pumas documentation)

    @derived begin
        # here is where we calculate concentration and add residual variability
        # tilde (~) means "distributed as"
        cp = @. 1000 * Central / Vc # ipred = A1/V
        dv ~ @. Normal(cp, σ)
        # dv ~ @. Normal(cp, sqrt(cp^2 * σ_prop^2 + σ_add^2))
    end
end

# Parameter values
params = (tvcl = 1.0, tvvc = 10.0, Ω = Diagonal([0.09, 0.09]), σ = 3.16)
params2 = (tvcl = 1.0, tvvc = 8.0, Ω = Diagonal([0.5, 0.5]), σ = 4.16)

# Fit a base model
fit_results = fit(model, population, params, Pumas.FOCE())
fit_results2 = fit(model, population, params, Pumas.NaivePooled(); omegas = (:Ω,))
fit_results3 = fit(model, population, params2, Pumas.LaplaceI())
fit_results4 = fit(model, population, params2, Pumas.FOCE(); constantcoef = (tvcl = 1.0,))

fit_compare = compare_estimates(;
    FOCE = fit_results,
    LaplaceI = fit_results3,
    NaivePooled = fit_results2,
    FOCE_constantcoef = fit_results4,
)

# VPCs
fit_vpc = vpc(fit_results) # Single-Threaded
fit_vpc = vpc(
    fit_results; # Multi-Threaded
    ensemblealg = EnsembleThreads(),
    prediction_correction = true, # pc-vpc
)

vpc_plot(fit_vpc)

# Generate a report for all of our fitted models
report((;
    FOCE = (fit_results, fit_vpc), # using vpc result
    LaplaceI = fit_results3,
    NaivePooled = fit_results2,
    FOCE_constantcoef = fit_results4,
))