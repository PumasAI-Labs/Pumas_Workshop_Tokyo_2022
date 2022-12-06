## STILL IN BETA ##

using DeepPumas
using StableRNGs
using PumasPlots
using CairoMakie
using Serialization

#=
Let's take DeepPumas out for a quick spin. To do that, we need some data where we know
precisely what the underlying dynamics are, what can be explained by covariates, and what
can not. So, we define a data-generating model from which we generate synthetic data.
=#
datamodel = @model begin
  @param begin
    tvImax ∈ RealDomain(; lower=0., init=1.1)
    tvIC50 ∈ RealDomain(; lower=0., init=0.8)
    tvKa ∈ RealDomain(; lower=0.)
    Ω ∈ PDiagDomain(; init = Diagonal([0.1, 0.1, 0.1]))
    σ ∈ RealDomain(; lower=0., init=0.08)
  end
  @random η ~ MvNormal(Ω)
  @covariates AGE Weight
  @pre begin
    Ka = tvKa * exp(η[1]) + 0.5 * (AGE/55)^2 
    Imax = tvImax * exp(η[2]) + 1.6*(Weight + AGE)/Weight 
    IC50 = tvIC50 * exp((Weight/75)^2 + η[3])
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - Imax * Central / (IC50 + Central)
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end

pop = synthetic_data(
  datamodel,
  DosageRegimen(4.);
  obstimes=0:0.3:5,
  nsubj=132,
  covariates=(;
    AGE=truncated(Normal(55,10), 15, Inf),
    Weight=truncated(Normal(75,25), 20, Inf),
  ),
  rng = StableRNG(123)
)

# It's generally a good idea to split your dataset into a train and a test set. 
trainpop = pop[1:120]
testpop = pop[121:end]

# Let's peek at the generated data along with the preds and ipreds of the data-generating
# model.
obstimes = 0:0.05:5 
true_pred = predict(datamodel, testpop, init_params(datamodel); obstimes) 
plotgrid(true_pred; pred=(; label="Best possible pred"))

#=
Now that we have some synthetic data, let's pretend we have no idea how the drug is cleared and
how the covariates affect patient outcomes. The traditional way of figuring this out would
be some combination of the "intuition" a scientist builds up over time, from which we can
make educated guesses based on the observed outcomes and some modeling trial and error.
Truthfully, though, I don't think I'd ever get that covariate effect by this approach.
The DeepPumas way is to rely on a combination of using neural networks and random effects to
capture terms and individual differences. There are many details here, but let's just run
through an example workflow and then we'll get back to the individual pieces and explain
them in more detail.
=#

model = @model begin
  @param begin
    tvKa ∈ RealDomain(; lower=0.)
    NN ∈ MLP(4, 4, 4, (1, identity); reg=L2(1.)) # Normal(0.0, 1/sqrt(2))
    # NN ∈ MLP(5, 4, 4, (1, identity); reg=L1(1.))
    ω_ka ∈ RealDomain(lower=0.) # optional: ω_ka ~ LogNormal()
    σ ∈ RealDomain(; lower=0., init=0.08)
  end
  @random begin
    η_ka ~ Normal(0., ω_ka)
    η1 ~ Normal()
    η2 ~ Normal()
    # η3 ~ Normal()
  end
  @pre begin
    Ka = tvKa * exp(η_ka)
    iNN = fix(NN, η1, η2)
    # iNN = fix(NN, η1, η2, η3)
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - iNN(Central, Depot)[1] # equivalent to NN(η1, η2, Central, Depot)
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end

fpm = fit(
  model,
  trainpop,
  init_params(model),
  MAP(FOCE()); 
  optim_options = (; iterations=200),
  diffeq_options = (; alg=Rodas5P())
)

## Note that we're predicting test data that was never used for fitting.
model_pred = predict(model, testpop, coef(fpm); obstimes);
plotgrid(model_pred)

using Pumas.ForwardDiff: jacobian
# DO NOT TRY THIS AT HOME
mean(1:500) do _
  jacobian(coef(fpm).NN, rand(4))
end

function sparsify(s::Subject, N)
  inds = unique([1; sort(rand(1:length(s.time), N - 1))])
  mask = falses(length(s.time))
  mask[inds] .= true
  o1 = NamedTuple{keys(s.observations)}(getindex.(values(s.observations), Ref(mask)))
  c = s.covariates
  t1 = s.time[mask]
  ev1 = s.events[minimum(t1) .≤ getfield.(s.events, :time) .≤ maximum(t1)]
  return Subject(s.id, o1, c, ev1, t1)
end
function sparsify(_pop::Population, N) 
  return map(Base.Fix2(sparsify, N), _pop)
end

sparse_testpop = sparsify(testpop, 5)
model_pred = predict(model, sparse_testpop, coef(fpm); obstimes);

plotgrid(true_pred; pred = (; label="Best possible pred", color=(:black, 0.5)), ipred = false)
plotgrid!(model_pred; pred = (; linestyle=:dash))

# Define a target mapping from patient data to the posterior random effects
target = preprocess(fpm)

# Define a mulit-layer perceptron (MLP) which is a simple kind of neural network.
nn = MLP(numinputs(target), 6, 6, (numoutputs(target), identity))

# Fit the neural network using different regularization and keep the fit that generalize
# best
fitted_nn = hyperopt(nn, target)

# DO NOT TRY THIS AT HOME
mean(1:120) do i
  jacobian(fitted_nn.ml.ml, target.x[:,i])
end

# Augment the original fitted pumas model with the fitted neural network.
covariate_model = augment(fpm, fitted_nn)

# The `init_params` of an augmented model is the combination of the best parameters of the
# FittedPumasModel and the fitted machine learning model.
p_covs = init_params(covariate_model);

sparse_testpop = sparsify(testpop, 5)
covariate_pred = predict(covariate_model, sparse_testpop, p_covs; obstimes);

plotgrid(true_pred; pred = (; label="Best possible pred", color=(:black, 0.5)), ipred = false)
plotgrid!(covariate_pred; pred = (; linestyle=:dash))

# Here, we see that the ipreds of the NN-embedded model match the observations almost
# perfectly. But much more impressively, the augmented `covariate_model` does not match the
# observations, but rather the preds of the data-generating model - the ground truth! The
# ipreds are overfitted. The machine learning-augmented, neural-embedded model is not. It
# has found the right relationships and can use them to predict the test data accurately.

# The above workflow can already produce nice fits, but it might never-the-less be improved.
# The "augment workflow"  is great, but it connects the ML fitting to the loglikelihood of
# the patient outcomes via a Normal approximation of the posterior distribution of the
# random effects. By proceeding with another stage of fitting, we can improve that
# relationship and get better fits. This is especially useful if there are any linear
# dependencies of the random effects.

deep_fpm = fit(
  covariate_model,
  trainpop,
  p_covs,
  MAP(FOCE()); 
  ## Let's not run for the hour or so we might need to find the optima.
  optim_options = (; time_limit = 5 * 60),
  diffeq_options = (; alg = Rodas5P())
)

# Now we've really fitted the full model, machine learning and all, towards the (FOCE
# approximated) loglikelihood that the longitudinal patient observations are samples of our
# model.

sparse_testpop = sparsify(testpop, 5)

deep_pred = predict(covariate_model, sparse_testpop, coef(deep_fpm); obstimes);

plotgrid(true_pred; pred = (; label="Ground truth", color=(:black, 0.5)), ipred = false)
plotgrid!(deep_pred; pred=(; linestyle=:dash))

coef(deep_fpm).ω_ka
coef(fpm).ω_ka
