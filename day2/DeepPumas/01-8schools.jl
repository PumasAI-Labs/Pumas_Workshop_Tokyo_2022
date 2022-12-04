using Pumas

model = @model begin
	@param begin
	  μ ∈ RealDomain()
	  τ ∈ RealDomain(; lower = 0.0)
	end
	@random η ~ Normal(μ, τ)
	@covariates σ
	@pre begin
        _η = η
        _σ = σ
    end
	@derived y ~ @. Normal(_η, _σ)
end

pop_y = [28.0,  8.0, -3.0, 7.0, -1.0,  1.0, 18.0, 12.0]
pop_σ = [14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6]
nsubj = length(pop_y)

pop = map(1:nsubj) do i
    Subject(
        id = i,
        covariates = (σ = pop_σ[i],),
        time = [0.0],
        observations = (y = [pop_y[i]],),
    )
end

iparam = (μ = 1.0, τ = 1.0)
r1 = fit(model, pop, iparam, LaplaceI())
r2 = fit(model, pop, iparam, JointMAP())

empirical_bayes(r1)
empirical_bayes(r2)
getindex.(empirical_bayes(r2), :η) - pop_y
