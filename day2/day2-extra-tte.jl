using Pumas
using PharmaDatasets
using Pumas, DataFramesMeta

# dataset
tte_single = dataset("tte_single")

# Pumas Modeling
pop_single = read_pumas(
    tte_single;
    observations = [:DV],
    covariates = [:DOSE],
    id = :ID,
    time = :TIME,
    event_data = false,
)

# Weibull
tte_single_weibull_model = @model begin
    @param begin
        λ₁ ∈ RealDomain(; lower = 0, init = 0.001) # basal hazard
        β ∈ RealDomain(; init = 0.001)           # fixed effect DOSE
        ω ∈ RealDomain(; lower = 0.001)          # inter-subject variability
        Κ ∈ RealDomain(; lower = 0, init = 0.001)  # Weibull shape
    end

    @random begin
        η ~ Normal(0.0, ω)
    end

    @covariates DOSE

    @pre begin
        _Κ = Κ                     # shape parameter
        _λ₁ = λ₁ * exp(η)          # basal hazard with inter-subject variability
        _λ₀ = _λ₁ * exp(β * DOSE)  # total hazard
    end

    @vars begin
        # Weibull
        # 1e-10 for model numerical stability
        λ = _λ₀ * _Κ * (_λ₀ * t + 1e-10)^(_Κ - 1)
    end

    @dynamics begin
        # the derivative of cumulative hazard is equal to hazard
        Λ' = λ
    end

    @derived begin
        DV ~ @. TimeToEvent(λ, Λ)
    end
end

tte_single_weibull_fit = fit(
    tte_single_weibull_model,
    pop_single,
    init_params(tte_single_weibull_model),
    LaplaceI(),
)

# Simulations
# First get CIs
tte_single_weibull_infer = infer(tte_single_weibull_fit)

# Simulation a TTE datasets is really simple
sims = simobstte(
    tte_single_weibull_model,
    pop_single,
    coef(tte_single_weibull_infer);
    maxT=500.0, # the censoring time
)

# Convert simulated population to a DataFrame for easier display
sims_df = DataFrame(sims)
