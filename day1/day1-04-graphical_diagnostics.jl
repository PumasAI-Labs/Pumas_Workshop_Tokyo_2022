inspect_covmodel = inspect(pkfit_base_wt_crcl)

# Covariate Analysis
## covariates check
covariates_check(pkfit_base_wt_crcl) # can also pass fit or inspect
## covariates distribution
covariates_dist(pkfit_base_wt_crcl) # can also pass fit or inspect

# covariate correlation
cc = @chain DataFrame(pop) begin
    unique(:id)
    select(:WT, :AGE, :CRCL)
    dropmissing
end
pp = PlottingUtilities.pair_plot(cc)

# model convergence
convergence_trace(inspect_covmodel)

# Goodness of Fit plots
goodness_of_fit(inspect_covmodel)
observations_vs_predictions(inspect_covmodel)
observations_vs_ipredictions(inspect_covmodel)

# using custom predict
fit_predict = predict(pkfit_base_wt_crcl)
observations_vs_ipredictions(pkfit_base_wt_crcl.model, fit_predict) # using model + custom predictions

# individual plots
sf = subject_fits(
    inspect_covmodel;
    separate = true,
    paginate = true,
    facet = (combinelabels = true,),
    figure = (resolution = (1400, 1000), fontsize = 28),
    axis = (ylabel = "Observed/Predicted drugY (ng/mL)",),
)

sf[1]

## richer plots
ind_preds = [
    predict(
        pkfit_base_wt_crcl.model,
        pkfit_base_wt_crcl.data[i],
        pkfit_base_wt_crcl.param,
        obstimes = minimum(pkfit_base_wt_crcl.data[i].time):1:maximum(
            pkfit_base_wt_crcl.data[i].time,
        ),
    ) for i = 1:length(pkfit_base_wt_crcl.data)
]
indplots = subject_fits(
    ind_preds,
    limit = 9,
    cols = 3,
    rows = 3,
    paginate = true,
    separate = true,
    facet = (combinelabels = true,),
)
#
[
    figurelegend(
        indplots[i];
        position = :t,
        alignmode = Outside(),
        orientation = :horizontal,
    ) for i = 1:length(indplots)
]

###########################
indplots[1]

# residual plots
# population residuals vs time
wresiduals_vs_time(inspect_covmodel)

# using custom likelihood approx
fit_wres = wresiduals(pkfit_base_wt_crcl, Pumas.FO())
wresiduals_vs_time(pkfit_base_wt_crcl.model, fit_wres) # using model + custom wresiduals

# individual residuals vs time
iwresiduals_vs_time(inspect_covmodel)

# population residuals vs population predictions
wresiduals_vs_predictions(inspect_covmodel)

# individual residuals vs individual predictions
# using custom likelihood approx + custom predict
iwresiduals_vs_ipredictions(inspect_covmodel)

# distribution of residuals
wresiduals_dist(inspect_covmodel)

# residuals vs covariates
wresiduals_vs_covariates(inspect_covmodel, figure = (resolution = (1400, 800),))


# Random effects plots
#distribution
empirical_bayes_dist(inspect_covmodel)

#ebe vs covariates
empirical_bayes_vs_covariates(inspect_covmodel, figure = (resolution = (1400, 800),))

# ebe correlation
ebes = @chain DataFrame(inspect_covmodel) begin
    select(r"^η")
    dropmissing
end
pp = PlottingUtilities.pair_plot(ebes)

# VPCs
## regular vpc
fit_vpc = vpc(
    pkfit_base_wt_crcl;
    ensemblealg = EnsembleThreads(), # Multi-Threaded
)
vpc_plot(fit_vpc)

# prediction corrected vpc
fit_pcvpc = vpc(
    pkfit_base_wt_crcl;
    ensemblealg = EnsembleThreads(), # Multi-Threaded
    prediction_correction = true, # pc-vpc
)
vpc_plot(fit_pcvpc)

# regular vpc with a differen x-axis
fit_wtvpc = vpc(
    pkfit_base_wt_crcl;
    covariates = [:WT],
    ensemblealg = EnsembleThreads(), # Multi-Threaded
)
vpc_plot(fit_wtvpc)

## changing the covariate in the X-axis and stratifying
fit_vpc_crcl = vpc(
    pkfit_base_wt_crcl;
    ensemblealg = EnsembleThreads(), # Multi-Threaded
    covariates = [:CRCL],
    stratify_by = [:SEX],
)
vpc_plot(fit_vpc_crcl)
