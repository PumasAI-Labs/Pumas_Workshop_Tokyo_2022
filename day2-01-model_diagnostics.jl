# Model diagnostics high level overview!
# More information at:
# - https://docs.pumas.ai/stable/analysis/plots/
# - https://docs.pumas.ai/stable/analysis/inference/
# - https://docs.pumas.ai/stable/analysis/apps/
# - https://docs.pumas.ai/stable/analysis/vpc/
#
# ... and tutorials, videos, etc!

# Load the necessary Libraries
using Pumas
using PumasUtilities
using PharmaDatasets
using CSV
using Random

# Read in data
# iv infusion given over 2 hours with demographic information (age, weight, sex, crcl)
pkfile = dataset("nlme_sample.csv", String)
pkdata = CSV.read(pkfile, DataFrame; missingstring = ["NA", ""])

pop = read_pumas(
    pkdata,
    id = :ID,
    time = :TIME,
    amt = :AMT,
    covariates = [:WT, :AGE, :SEX, :CRCL, :GROUP],
    observations = [:DV],
    cmt = :CMT,
    evid = :EVID,
    rate = :RATE,
)

# 2-cpt with weight and CRCL
model = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0)
        tvvc ∈ RealDomain(lower = 0)
        tvq ∈ RealDomain(lower = 0)
        tvvp ∈ RealDomain(lower = 0)
        allometric ∈ RealDomain()
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates WT CRCL

    @pre begin
        wtCL = (WT / 70)
        wtV = (WT / 70)
        crcl_eff = (CRCL / 95)
        CL = tvcl * (crcl_eff * wtCL)^allometric * exp(η[1])
        Vc = tvvc * wtV^allometric * exp(η[2])
        Q = tvq
        Vp = tvvp
    end

    @dynamics Central1Periph1

    @derived begin
        CONC := @. Central / Vc
        DV ~ @. Normal(CONC, σ)
    end
end

# Before moving on
# Covariate Analysis
## covariates check
covariates_check(model, pop) # can also pass fit or inspect
covariates_dist(model, pop) # can also pass fit or inspect

# Parameter values
params =
    (tvvc = 5, tvcl = 0.02, tvq = 0.01, tvvp = 10, allometric = 1.0, Ω = Diagonal([0.01, 0.01]), σ = 0.01)

# Check for finite loglikelihood
loglikelihood(model, pop, params, Pumas.FOCE()) # using model + population + params + estimation method
findinfluential(model, pop, params, Pumas.FOCE(); k = 30)

# Fit model
fit_results_base = fit(model, pop, params, Pumas.FOCE(); constantcoef=(allometric=0.0,))

# Model metrics in a DataFrame
fit_results_base_metrics = metrics_table(fit_results_base)

# Loglikelihood and NONMEM's OFV with constant
loglikelihood(fit_results_base) # using a result from fit

# to reproduce NONMEM's "OFV with constant" you need to divide by -2
loglikelihood(fit_results_base) / -2

# AIC
aic(fit_results_base)

# BIC
bic(fit_results_base)

# ϵ-Shinkrage
ϵshrinkage(fit_results_base)

#η-Shinkrage
ηshrinkage(fit_results_base)

# Inspect the models
# this runs predictions, wres, ebes, icoef and dosecontrol
fit_inspect_base = inspect(fit_results_base)

# Open the app
inspect_app = evaluate_diagnostics(fit_inspect_base)

# output in a handy DataFrame format
fit_inspect_df = DataFrame(fit_inspect_base)
# too many columns... workspace browser...

# you can get any of the single results using the functions
DataFrame(predict(fit_results_base))
DataFrame(wresiduals(fit_results_base))
DataFrame(icoef(fit_results_base))

# Plot individual parts

## Goodness of Fit plots
goodness_of_fit(fit_inspect_base)
observations_vs_predictions(fit_inspect_base)
observations_vs_ipredictions(fit_inspect_base)
wresiduals_vs_time(fit_inspect_base)
iwresiduals_vs_ipredictions(fit_inspect_base)

empirical_bayes_vs_covariates(fit_inspect_base)
interactive(empirical_bayes_vs_covariates(fit_inspect_base))

# Setting options for plots
empirical_bayes_vs_covariates(fit_inspect_base;
    figure = (fontsize=11,),
    color=:red)

empirical_bayes_vs_covariates(fit_inspect_base;
    figure = (fontsize=11,),
    color=:red,
    markersize=5,)

# Fit allometric scaling model
fit_results_allometric = fit(model, pop, params, Pumas.FOCE(); constantcoef=(allometric=1.0,))
compare_estimates(;fit_results_base, fit_results_allometric)

fit_inspect_allometric = inspect(fit_results_allometric)

goodness_of_fit(fit_inspect_allometric)
observations_vs_predictions(fit_inspect_allometric)
empirical_bayes_vs_covariates(fit_inspect_allometric)

# Inference
## Confidence Intervals using asymptotic variance-covariance
fit_infer = infer(fit_results_allometric)
coeftable(fit_results_allometric)  # DataFrame
stderror(fit_infer)

## Confidence Intervals using bootstrap
fit_infer_bs = infer(fit_results_allometric, Pumas.Bootstrap(samples = 100))
coeftable(fit_infer_bs)
stderror(fit_infer)
DataFrame(fit_infer_bs.vcov)

## Stratified sampling of subjects
fit_infer_bs_stratify = infer(fit_results_allometric, Pumas.Bootstrap(samples = 100, stratify_by=:WT))
coeftable(fit_infer_bs_stratify)
stderror(fit_infer)

# Seed
rng = Random.seed!(2131)
fit_infer_bs_rng_1 = infer(fit_results_allometric, Pumas.Bootstrap(;samples = 20, rng))
rng = Random.seed!(2131)
fit_infer_bs_rng_2 = infer(fit_results_allometric, Pumas.Bootstrap(;samples = 20, rng))
coeftable(fit_infer_bs_rng_1)
coeftable(fit_infer_bs_rng_2)

## Confidence Intervals using SIR
fit_infer_sir = infer(fit_results_allometric, Pumas.SIR(samples = 100, resamples = 10))
coeftable(fit_infer_sir)

# VPCs
## default covariate in the X-axis is time
fit_vpc = vpc(fit_results_allometric) # Single-Threaded
fit_vpc = vpc(
    fit_results_allometric;
    bandwidth = 30,
    prediction_correction = false, # pc-vpc
)

vpc_fig = vpc_plot(fit_vpc)
figurelegend(vpc_fig)
vpc_fig

## stratification
fit_vpc_allometric = vpc(
    fit_results_allometric;
    stratify_by = [:SEX],
)
vpc_plot(fit_vpc_allometric)

## changing the independent variable
fit_vpc_allometric = vpc(
    fit_results_allometric;
    covariates = [:CRCL],
)
vpc_plot(fit_vpc_allometric)