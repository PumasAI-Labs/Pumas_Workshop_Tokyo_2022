using Pumas
using PumasUtilities
using Serialization

pkfit_base_wt_crcl = ("./day1/pkfit_base_wt_crcl.jls")

# A fitted model stores the model, data and parameters
# we can easily access them
orig_model = pkfit_base_wt_crcl.model
original_pop = pkfit_base_wt_crcl.data
fitted_params = pkfit_base_wt_crcl.param

# perform a simple simulation from the fitted model
sims = simobs(pkfit_base_wt_crcl)
sim_plot(sims)

# perform a simple simulation from the model, population and parameters
sims1 = simobs(orig_model, original_pop, fitted_params)
sim_plot(sims1)

# perform a simple simulation from the model, population and parameters and rich time grid
sims2 = simobs(orig_model, original_pop, fitted_params, obstimes = 0:1:196)
sim_plot(sims2)

# perform the above simulation without error term
sims3 = simobs(
    orig_model,
    original_pop,
    fitted_params,
    obstimes = 0:1:196,
    simulate_error = false,
)
sim_plot(sims3)


# perform the above simulation without error term but change the dose
sims4 = simobs(
    orig_model,
    Subject.(original_pop, events = DosageRegimen(750, addl = 7, ii = 24)),
    fitted_params,
    obstimes = 0:1:196,
    simulate_error = false,
)
sim_plot(sims4)

# perform the above simulation without error term but change the dose
# and simulate using empirical_bayes estimates
sims5 = simobs(
    orig_model,
    Subject.(original_pop, events = DosageRegimen(750, addl = 7, ii = 24)),
    fitted_params,
    empirical_bayes(pkfit_base_wt_crcl),
    obstimes = 0:1:196,
    simulate_error = false,
)
sim_plot(sims5)

## Exporting a simulation
DataFrame(sims5)

# perform the above simulation without error term but change the dose
# and simulate without any random effects on subject level
sims6 = simobs(
    orig_model,
    Subject.(original_pop, events = DosageRegimen(750, addl = 7, ii = 24)),
    (fitted_params..., σ²_add = 0.0, σ²_prop = 0.0),
    zero_randeffs(orig_model, original_pop, fitted_params),
    obstimes = 0:1:196,
    simulate_error = false,
)
sim_plot(sims6)

## Exporting a simulation
DataFrame(sims5)


## Repeating a simulation many times

# single replication
sim_1rep = simobs(
    orig_model,
    original_pop,
    fitted_params,
    obstimes = 0:1:196,
    simulate_error = false,
)
sim_1rep_df = DataFrame(sim_1rep)

# 10 replication
sim_10rep = [
    simobs(
        orig_model,
        original_pop,
        fitted_params,
        obstimes = 0:1:196,
        simulate_error = false,
    ) for i = 1:10
]
sim_10rep_df = reduce(vcat, DataFrame.(sim_10rep), source = "rep")

# simulate from uncertainty distributions
infer_covmodel = infer(pkfit_base_wt_crcl)
simvcov = simobs(infer_covmodel, samples = 20)
simvcov_df = reduce(vcat, DataFrame.(simvcov), source = "rep")
## pass a uncertainty simulation object into the vpc
vpcvcov = vpc(simvcov)
vpc_plot(vpcvcov)

infer_covmodel_bs = infer(pkfit_base_wt_crcl, Pumas.Bootstrap(; samples = 20))
simbs = simobs(infer_covmodel, samples = 20)
simbs_df = reduce(vcat, DataFrame.(simbs), source = "rep")
## pass a uncertainty simulation object into the vpc
vpcbs = vpc(simbs)
vpc_plot(vpcbs)

# get the standard errors and parameteric confidence intervals using SIR
infer_covmodel_sir = infer(pkfit_base_wt_crcl, Pumas.SIR(; samples = 20, resamples = 5))
simsir = simobs(infer_covmodel, samples = 20)
simsir_df = reduce(vcat, DataFrame.(simsir), source = "rep")
## pass a uncertainty simulation object into the vpc
vpcsir = vpc(simsir)
vpc_plot(vpcsir)

## generate a new population and simulate into it



sims7 = simobs(orig_model, s1, fitted_params)
sim_plot(sims7)


