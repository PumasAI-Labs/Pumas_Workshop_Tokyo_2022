### Demonstrate non-Gaussian distributions in Pumas ###

using Pumas, PumasUtilities, QuadGK, PharmaDatasets, Random, AlgebraOfGraphics, DataFrames

# Mean of LogNormal and Gamma
μPK = 1.0
σ = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

# Compare means and coefficient of variation as a function of σ for LogNormal and Gamma
DataFrame(
    μ = μPK,
    σ = σ,
    meanLogNormal = mean.(LogNormal.(log.(μPK), σ)),
    cvLogNormal = std.(LogNormal.(log.(μPK), σ)) ./ mean.(LogNormal.(log.(μPK), σ)),
    meanGamma = mean.(Gamma.(1 ./ σ .^ 2, μPK .* σ .^ 2)),
    cvGamma = std.(Gamma.(1 ./ σ .^ 2, μPK .* σ .^ 2)) ./
              mean.(Gamma.(1 ./ σ .^ 2, μPK .* σ .^ 2)),
)

# Mean of LogitNormal and Beta
μbioav = 0.7
DataFrame(
    μ = μbioav,
    σ = σ,
    meanLogitNormal = map(
        _σ -> quadgk(
            t -> logistic(t) * pdf(Normal(logit(μbioav), _σ), t),
            -100 * _σ,
            100 * _σ,
        )[1],
        σ,
    ),
    meanBeta = mean.(Beta.(μbioav ./ σ, (1 - μbioav) ./ σ)),
)

# Fitting with LogNormal and Gamma

# Fetch the warfarin data
pkdata = read_pumas(dataset("pumas/warfarin"))

# define a traditional one compartment model immediate absorption
# where individual parameters are log-normally distributed
mdlLogNormal = @model begin
    @param begin
        θCL ∈ RealDomain(lower = 0.0)
        θVc ∈ RealDomain(lower = 0.0)

        ωCL ∈ RealDomain(lower = 0.0)
        ωVc ∈ RealDomain(lower = 0.0)

        σ ∈ RealDomain(lower = 0.0)
    end
    @random begin
        _CL ~ LogNormal(log(θCL), ωCL)
        _Vc ~ LogNormal(log(θVc), ωVc)
    end
    # This is equivalent to defining
    # CL = θCL*exp(ηCL)
    # with
    # ηCL = Normal(0.0, ωCL)
    @pre begin
        CL = _CL
        Vc = _Vc
    end
    @dynamics Central1
    @derived begin
        μ := Central / Vc
        dv ~ @. Normal(μ, abs(μ) * σ)
    end
end

# Define a similar model but using the Gamma distribution for the individual parameters
mdlGamma = @model begin
    @param begin
        θCL ∈ RealDomain(lower = 0.0)
        θVc ∈ RealDomain(lower = 0.0)

        ωCL ∈ RealDomain(lower = 0.0)
        ωVc ∈ RealDomain(lower = 0.0)

        σ ∈ RealDomain(lower = 0.0)
    end
    @random begin
        # See https://docs.pumas.ai/stable/model_components/error_models/#Gamma
        _CL ~ Gamma(1 / ωCL^2, θCL * ωCL^2)
        _Vc ~ Gamma(1 / ωVc^2, θVc * ωVc^2)
    end
    @pre begin
        CL = _CL
        Vc = _Vc
    end
    @dynamics Central1
    @derived begin
        μ := Central / Vc
        dv ~ @. Normal(μ, abs(μ) * σ)
    end
end

# Define inital values for the fitting
initparam = (θCL = 1.0, θVc = 5.0, ωCL = 0.1, ωVc = 0.1, σ = 0.2)

# fit the two models
fitLogNormal = fit(mdlLogNormal, pkdata, initparam, FOCE())
fitGamma = fit(mdlGamma, pkdata, initparam, FOCE())

# and compare the estimates
compare_estimates(; fitLogNormal, fitGamma)

# as mention above, the mean of a LogNormal is exp(μ + σ²/2)
# so let's compare that with the Gamma typical values
DataFrame(
    θLogNormal = [coef(fitLogNormal).θCL, coef(fitLogNormal).θVc],
    ELogNormal = [
        exp(log(coef(fitLogNormal).θCL) + coef(fitLogNormal).ωCL^2 / 2),
        exp(log(coef(fitLogNormal).θVc) + coef(fitLogNormal).ωVc^2 / 2),
    ],
    θGamma = [coef(fitGamma).θCL, coef(fitGamma).θVc],
)

# Data a DataFrame with the plotting data for the two PDFs
plotdataPK = DataFrame(x = range(0, 0.5, length = 1000))
plotdataPK = transform(
    plotdataPK,
    "x" =>
        ByRow(
            t -> pdf(LogNormal(log(coef(fitLogNormal).θCL), coef(fitLogNormal).ωCL), t),
        ) => "lognormal",
    "x" =>
        ByRow(
            t -> pdf(
                Gamma(1 / coef(fitGamma).ωCL^2, coef(fitGamma).θCL * coef(fitGamma).ωCL^2),
                t,
            ),
        ) => "gamma",
)

# plot the PDFs with AlgebraOfGraphics
data(stack(plotdataPK, ["lognormal", "gamma"])) *
mapping("x", "value", color = "variable") *
visual(AlgebraOfGraphics.Lines) |> draw



# Fitting with LogitNormal and Beta

# For illustrating a Beta distributed random effect, we'll use a dual absorption model with
# relative bioavailability

# First, a traditional LogitNormal version
mdlLogitNormal = @model begin
    @param begin
        θkaFast ∈ RealDomain(lower = 0.0)
        θkaSlow ∈ RealDomain(lower = 0.0)
        θCL ∈ RealDomain(lower = 0.0)
        θVc ∈ RealDomain(lower = 0.0)
        # Note the init
        θbioav ∈ RealDomain(lower = 0.0, init = 0.7)

        ωCL ∈ RealDomain(lower = 0.0)
        ωVc ∈ RealDomain(lower = 0.0)
        # I call this one ξ to distinguish it from ω since the interpretation is NOT a relative error (coefficient of variation)
        # Note the init
        ξbioav ∈ RealDomain(lower = 0.0, init = 0.1)

        σ ∈ RealDomain(lower = 0.0)
    end
    @random begin
        _CL ~ Gamma(1 / ωCL^2, θCL * ωCL^2)
        _Vc ~ Gamma(1 / ωVc^2, θVc * ωVc^2)
        # define the latent Gaussian random effect. Notice the logit transform
        _bioavLogit ~ Normal(logit(θbioav), ξbioav)
    end
    @pre begin
        kaFast = θkaFast
        kaSlow = θkaSlow
        CL = _CL
        Vc = _Vc
    end
    @dosecontrol begin
        # _bioav is LogitNormal distributed
        _bioav = logistic(_bioavLogit)
        bioav = (DepotFast = _bioav, DepotSlow = 1 - _bioav)
    end
    @dynamics begin
        DepotFast' = -kaFast * DepotFast
        DepotSlow' = -kaSlow * DepotSlow
        Central' = kaFast * DepotFast + kaSlow * DepotSlow - CL / Vc * Central
    end
    @derived begin
        μ := Central / Vc
        dv ~ @. Normal(μ, abs(μ) * σ)
    end
end

# next, a similar model but using a Beta distribution instead of LogitNormal
mdlBeta = @model begin
    @param begin
        θkaFast ∈ RealDomain(lower = 0.0)
        θkaSlow ∈ RealDomain(lower = 0.0)
        θCL ∈ RealDomain(lower = 0.0)
        θVc ∈ RealDomain(lower = 0.0)
        # Note the init
        θbioav ∈ RealDomain(lower = 0.0, init = 0.7)

        ωCL ∈ RealDomain(lower = 0.0)
        ωVc ∈ RealDomain(lower = 0.0)
        # We call this one n since the interpretation is like the length of a Binomial distribution
        # Note the init
        nbioav ∈ RealDomain(lower = 0.0, init = 10.0)

        σ ∈ RealDomain(lower = 0.0)
    end
    @random begin
        _CL ~ Gamma(1 / ωCL^2, θCL * ωCL^2)
        _Vc ~ Gamma(1 / ωVc^2, θVc * ωVc^2)
        # The makes E(_bioav) = θbioav
        # See https://en.wikipedia.org/wiki/Beta_distribution
        _bioav ~ Beta(θbioav * nbioav, (1 - θbioav) * nbioav)
    end
    @pre begin
        kaFast = θkaFast
        kaSlow = θkaSlow
        CL = _CL
        Vc = _Vc
    end
    @dosecontrol begin
        bioav = (DepotFast = _bioav, DepotSlow = 1 - _bioav)
    end
    @dynamics begin
        DepotFast' = -kaFast * DepotFast
        DepotSlow' = -kaSlow * DepotSlow
        Central' = kaFast * DepotFast + kaSlow * DepotSlow - CL / Vc * Central
    end
    @derived begin
        μ := Central / Vc
        dv ~ @. Normal(μ, abs(μ) * σ)
    end
end

# Now simulate some data to use for the simulattion
#
# Define doses in the two depot compartments
dr = DosageRegimen(
    DosageRegimen(100, cmt = :DepotFast),
    DosageRegimen(100, cmt = :DepotSlow),
)
# Define sample times
simtimes = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0]
# Define the true parameters
trueparam = (
    θkaFast = 0.9,
    θkaSlow = 0.2,
    θCL = 1.1,
    θVc = 10.0,
    # Note the init
    θbioav = 0.7,
    ωCL = 0.1,
    ωVc = 0.1,
    # Note the init
    nbioav = 40.0,
    σ = 0.1,
)

# For simplicity, we just add 20% to the true values for initial values
initparamBeta = map(t -> 1.2 * t, trueparam)
# The initial values for the LogitNormal need to have ξbioav defined instead of nbioav
initparamLogitNormal =
    (Base.structdiff(initparamBeta, NamedTuple{(:nbioav,)})..., ξbioav = 0.1)

# Setup empty subjects with the dose information
simpop = [Subject(id = i, events = dr) for i = 1:40]

# Set the seed for reprocibility
_rng = Random.seed!(Random.default_rng(), 128)

# Simulate the data
pop = Subject.(simobs(mdlBeta, simpop, trueparam; obstimes = simtimes, rng = _rng))

# fit the two models
fitBioavLogitNormal = fit(mdlLogitNormal, pop, initparamLogitNormal, FOCE())
fitBioavBeta = fit(mdlBeta, pop, initparamBeta, FOCE())

# compare the estimate to the true values
leftjoin(
    compare_estimates(; fitBioavLogitNormal, fitBioavBeta),
    DataFrame(
        parameter = String.([keys(trueparam)...]),
        trueparams = [values(trueparam)...],
    ),
    on = "parameter",
)

# now let's plot the two bioavailability distributions
#
# first define a DataFrame with the PDF data
plotdatabioav = DataFrame(x = range(0, 1, length = 1000))
plotdatabioav = transform(
    plotdatabioav,
    "x" =>
        ByRow(
            t ->
                1 / coef(fitBioavLogitNormal).ξbioav / √(2π) / (t * (1 - t)) * exp(
                    -(logit(t) - logit(coef(fitBioavLogitNormal).θbioav))^2 /
                    (2 * coef(fitBioavLogitNormal).ξbioav^2),
                ),
        ) => "logitnormal",
    "x" =>
        ByRow(
            t -> pdf(
                Beta(
                    coef(fitBioavBeta).θbioav * coef(fitBioavBeta).nbioav,
                    (1 - coef(fitBioavBeta).θbioav) * coef(fitBioavBeta).nbioav,
                ),
                t,
            ),
        ) => "beta",
)

# and use AlgebraOfGraphics for plotting
data(stack(plotdatabioav, ["logitnormal", "beta"])) *
mapping("x", "value", color = "variable") *
visual(AlgebraOfGraphics.Lines) |> draw

### A Gamma error model
## Let's look at the coefficient of variation
DataFrame(
    μ = μPK,
    σ = σ,
    cvProportionalNormal = std.(Normal.(μPK, μPK .* σ)) ./ mean.(Normal.(μPK, μPK .* σ)),
    cvGamma = std.(Gamma.(1 ./ σ .^ 2, μPK .* σ .^ 2)) ./
              mean.(Gamma.(1 ./ σ .^ 2, μPK .* σ .^ 2)),
)

# This models combines all the non-Gaussian components:
# - Random effects for clearance and volume are Gamma distributed
# - Random effect for relative bioavailability is Beta distributed
# - Error model is Gamma distributed
#
# Notice that the error model is specified through the conditional distribution
# of the observed variable given the random effects. I.e., no explicit evaluation
# of the log-likelihood. The same approach can be used for discrete models, such
# as Binomial/Bernoulli, Poisson, NegativeBinomial, Categorical
mdlFull = @model begin
    @param begin
        θkaFast ∈ RealDomain(lower = 0.0)
        θkaSlow ∈ RealDomain(lower = 0.0)
        θCL ∈ RealDomain(lower = 0.0)
        θVc ∈ RealDomain(lower = 0.0)
        # Note the init
        θbioav ∈ RealDomain(lower = 0.0, init = 0.7)

        ωCL ∈ RealDomain(lower = 0.0)
        ωVc ∈ RealDomain(lower = 0.0)
        # Note the init
        nbioav ∈ RealDomain(lower = 0.0, init = 10.0)

        σ ∈ RealDomain(lower = 0.0)
    end
    @random begin
        _CL ~ Gamma(1 / ωCL^2, θCL * ωCL^2)
        _Vc ~ Gamma(1 / ωVc^2, θVc * ωVc^2)
        _bioav ~ Beta(θbioav * nbioav, (1 - θbioav) * nbioav)
    end
    @pre begin
        kaFast = θkaFast
        kaSlow = θkaSlow
        CL = _CL
        Vc = _Vc
    end
    @dosecontrol begin
        bioav = (DepotFast = _bioav, DepotSlow = 1 - _bioav)
    end
    @dynamics begin
        DepotFast' = -kaFast * DepotFast
        DepotSlow' = -kaSlow * DepotSlow
        Central' = kaFast * DepotFast + kaSlow * DepotSlow - CL / Vc * Central
    end
    @derived begin
        μ := Central / Vc
        dv ~ @. Gamma(1 / σ^2, μ * σ^2)
    end
end

# Fit the completely non-Gaussian model
fitFull = fit(mdlFull, pop, initparamBeta, FOCE())

# Compare all the results
leftjoin(
    compare_estimates(; fitBioavLogitNormal, fitBioavBeta, fitFull),
    DataFrame(
        parameter = String.([keys(trueparam)...]),
        trueparams = [values(trueparam)...],
    ),
    on = "parameter",
)
