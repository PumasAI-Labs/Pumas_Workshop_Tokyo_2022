# Load the necessary Libraries
using Pumas
using PumasUtilities
using PharmaDatasets
using CSV

# Read in data
# iv infusion given over 2 hours with demographic information (age, weight, sex, crcl)
pkfile = dataset("nlme_sample.csv", String)
pkdata = CSV.read(pkfile, DataFrame; missingstrings = ["NA", ""])

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
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates WT CRCL

    @pre begin
        wtCL = (WT / 70)^0.75
        wtV = (WT / 70)
        crcl_eff = (CRCL / 95)^0.75
        CL = tvcl * crcl_eff * wtCL * exp(η[1])
        Vc = tvvc * wtV * exp(η[2])
        Q = tvq
        Vp = tvvp
    end

    @dynamics Central1Periph1

    @derived begin
        CONC := @. Central / Vc
        DV ~ @. Normal(CONC, σ)
    end
end

# Parameter values
params =
    (tvvc = 5, tvcl = 0.02, tvq = 0.01, tvvp = 10, Ω = Diagonal([0.01, 0.01]), σ = 0.01)

# Fit models
fit_results = fit(model, pop, params, Pumas.FOCE())
fit_results_naive = fit(model, pop, params, Pumas.NaivePooled(); omegas = (:Ω,))
fit_results_fixed = fit(model, pop, params, Pumas.FOCE(); constantcoef = (tvcl = 0.3,))

# Confidence Intervals using asymptotic variance-covariance
fit_infer = infer(fit_results)
coeftable(fit_infer)  # DataFrame

# Confidence Intervals using bootstrap
fit_infer_bs = infer(fit_results, Pumas.Bootstrap(samples = 100))
coeftable(fit_infer_bs)

# Confidence Intervals using SIR
fit_infer_sir = infer(fit_results, Pumas.SIR(samples = 10, resamples = 10))
coeftable(fit_infer_sir)

# Loglikelihood and NONMEM's OFV with constant
loglikelihood(fit_results) # using a result from fit
loglikelihood(model, pop, coef(fit_results), Pumas.FOCE()) # using model + population + params + estimation method
loglikelihood(model, pop[1], coef(fit_results), Pumas.FOCE()) # using model + subject + params + estimation method
# to reproduce NONMEM's "OFV with constant" you need to divide by -2
loglikelihood(fit_results) / -2

# AIC
aic(fit_results)
aic(fit_results_naive)
aic(fit_results_fixed)

# BIC
bic(fit_results)
bic(fit_results_naive)
bic(fit_results_fixed)

# ϵ-Shinkrage
ϵshrinkage(fit_results)
ϵshrinkage(fit_results_naive)
ϵshrinkage(fit_results_fixed)

# η-Shinkrage
ηshrinkage(fit_results)
ηshrinkage(fit_results_naive)
ηshrinkage(fit_results_fixed)

# Model metrics in a DataFrame
metrics_table(fit_results)
metrics_table(fit_results_naive)
metrics_table(fit_results_fixed)

# Get the individual coefficients as a DataFrame
DataFrame(icoef(fit_results))

# Inspect the models
# this runs predictions, wres, ebes, icoef and dosecontrol
fit_inspect = inspect(fit_results)
fit_inspect_naive = inspect(fit_results_naive)
fit_inspect_fixed = inspect(fit_results_fixed)

# output in a handy DataFrame format
DataFrame(fit_inspect)
DataFrame(fit_inspect_naive)
DataFrame(fit_inspect_fixed)

# you can get any of the single results using the functions
DataFrame(predict(fit_results))
DataFrame(wresiduals(fit_results))
DataFrame(empirical_bayes(fit_results))
DataFrame(icoef(fit_results))
DataFrame(dosecontrol(fit_results)) # no dose control parameters here