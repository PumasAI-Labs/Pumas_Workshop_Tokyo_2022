using Pumas
using PharmaDatasets

pain_remed = dataset("pumas/pain_remed")

# The variable is coded 0:3 but Categorical starts at 1
pain_remed[!, :painord] = pain_remed[!, :painord] .+ 1

pop = read_pumas(
    pain_remed,
    observations = [:painord],
    covariates = [:conc],
    event_data = false,
)

ordinal_saem = @emmodel begin
    @random begin
        b₁ ~ 1 | Normal
        b₂ ~ 1 | LogNormal
        b₃ ~ 1 | LogNormal
        slope ~ 1 | Normal
    end
    @covariates conc # time varying
    @pre begin
        effect = slope * conc

        #Logit of cumulative probabilities
        lge₀ = b₁ + effect
        lge₁ = lge₀ - b₂
        lge₂ = lge₁ - b₃

        #Probabilities of >=0 and >=1 and >=2
        pge₀ = logistic(lge₀)
        pge₁ = logistic(lge₁)
        pge₂ = logistic(lge₂)

        #Probabilities of Y=0,1,2,3
        p₀ = 1.0 - pge₀
        p₁ = pge₀ - pge₁
        p₂ = pge₁ - pge₂
        p₃ = pge₂
    end

    @error begin
        painord ~ Categorical(p₀, p₁, p₂, p₃)
    end
end

init_params = (; b₁ = 0.0, b₂ = 1.0, b₃ = 1.0, slope = 0.0)

ordinal_fit = fit(ordinal_saem, pop, init_params, Pumas.SAEM(iters = (2_000, 1_000, 1_000)))

# The workflow is the same as any Pumas model
# Confidence Intervals
infer(ordinal_fit)
# Inspect
inspect(ordinal_fit)

# To get the MCMC chains from each subject:
icoef(ordinal_fit)
