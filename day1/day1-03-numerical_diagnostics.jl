using Pumas
using PumasUtilities
using Serialization
using Latexify
using Random
using PharmaDatasets
using SummaryTables

## load the fitted model if it is not in your workspace
# pkfit_base_wt_crcl = deserialize("pkfit_base_wt_crcl.jls");

pkdata = dataset("nlme_sample.csv")
## generate a table of demographics
demotable = @chain pkdata begin
    unique(:ID)
    select(:ID, :GROUP, :SEX, :WT, :AGE, :CRCL)
    @rtransform :SEX = :SEX == "M" ? "Male" : "Female"
end

table_one(demotable, [:GROUP, :SEX, :WT, :AGE, :CRCL]; nonnormal=[:GROUP, :SEX])

table_one(demotable, [:SEX, :WT, :AGE, :CRCL]; groupby = [:GROUP], nonnormal=[:SEX])


## QC model code for final check
render(latexify(pkfit_base_wt_crcl.model, :dynamics))

# fit helper functions
# generates a namedtuple that can be used as initial estimates in a subsequent fit
coef(pkfit_base_wt_crcl)

# creates a dataframe of the fitted results
coeftable(pkfit_base_wt_crcl) 

# creates a dataframe of the fitted results with parameters descriptions if provided
coefficients_table(pkfit_base_wt_crcl) 

# generates a table of model metrics
metrics_table(pkfit_base_wt_crcl) 

# number of observations in the fit
nobs(pkfit_base_wt_crcl) 

# Loglikelihood and NONMEM's OFV with constant
loglikelihood(pkfit_base_wt_crcl) # using a result from fit

# to reproduce NONMEM's "OFV with constant" you need to multiply by -2
loglikelihood(pkfit_base_wt_crcl) * -2

# AIC
aic(pkfit_base_wt_crcl)

# BIC
bic(pkfit_base_wt_crcl)

# ϵ-Shinkrage
ϵshrinkage(pkfit_base_wt_crcl)

#η-Shinkrage
ηshrinkage(pkfit_base_wt_crcl)

#variance-covariance matrix
vcov(pkfit_base_wt_crcl)

# convert the variance-covariance to a correlation matrix
StatsBase.cov2cor(vcov(pkfit_base_wt_crcl))

#standard errors
stderror(pkfit_base_wt_crcl)


# make an inference using asymptotic covariance matrix, sandwich_estimator
infer_covmodel = infer(pkfit_base_wt_crcl)

# change confidence level
infer_covmodel_lv = infer(pkfit_base_wt_crcl, level=0.90)

# infer helper function
# generate a dataframe of the inferred results
coeftable(infer_covmodel)

# change the CI level in the reported dataframe
coeftable(infer_covmodel, level = 0.90)


# acccess the variance covariance of the inferred object
infer_covmodel.vcov

# convert the variance-covariance to a correlation matrix
StatsBase.cov2cor(infer_covmodel.vcov)

# make an inference using the hessian, set sandwich_estimator to false
infer_covmodel_noswch = infer(pkfit_base_wt_crcl; sandwich_estimator = false)

# get the standard errors and non-parameteric confidence intervals using bootstrap
infer_covmodel_bs = infer(pkfit_base_wt_crcl, Pumas.Bootstrap(; samples = 20))

# get the standard errors and parameteric confidence intervals using SIR
infer_covmodel_sir = infer(pkfit_base_wt_crcl, Pumas.SIR(; samples = 20, resamples = 5))

# inspect the fit with diagnostics by computing predictions, residuals, EBEs and individual parameters
inspect_covmodel = inspect(pkfit_base_wt_crcl)

# convert the inspected object into a dataframe
DataFrame(inspect_covmodel)

# trim down your output by removing sets of columns
DataFrame(
    inspect_covmodel;
    include_covariates = false, # include/remove covariates
    include_events = false, # include/remove dose related events
    include_icoef = false, # include/remove individual parameters
    include_observations = false, # include/remove observations
    include_dosecontrol = false,  # include/remove dose control parameters
) 

# compute residuals using a different likelihood approximation. FO gives us WRES
inspect_covmodel_FO = inspect(pkfit_base_wt_crcl; wres_approx = Pumas.FO())
DataFrame(inspect_covmodel_FO; 
    include_covariates = false, # include/remove covariates
    include_events = false, # include/remove dose related events
    include_icoef = false, # include/remove individual parameters
    include_observations = false, # include/remove observations
    include_dosecontrol = false,  # include/remove dose control parameters)
)

# compute normalized prediction distributed errors
inspect_covmodel_npde = inspect(pkfit_base_wt_crcl; nsim = 100, rng = Random.seed!(1234))
DataFrame(inspect_covmodel_npde; include_covariates = false, include_events = false)

# you can get any of the single results using the functions
DataFrame(predict(pkfit_base_wt_crcl))
DataFrame(wresiduals(pkfit_base_wt_crcl))
DataFrame(empirical_bayes(pkfit_base_wt_crcl))
Pumas.empirical_bayes_dist(pkfit_base_wt_crcl)
DataFrame(icoef(pkfit_base_wt_crcl))
DataFrame(dosecontrol(pkfit_base_wt_crcl)) # no dose control parameters here
