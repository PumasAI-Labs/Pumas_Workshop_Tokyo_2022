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
params = (
    tvvc = 5,
    tvcl = 0.02,
    tvq = 0.01,
    tvvp = 10,
    allometric = 1.0,
    Ω = Diagonal([0.01, 0.01]),
    σ = 0.01,
)

# Check for finite loglikelihood
loglikelihood(model, pop, params, Pumas.FOCE()) # using model + population + params + estimation method
findinfluential(model, pop, params, Pumas.FOCE(); k = 30)
