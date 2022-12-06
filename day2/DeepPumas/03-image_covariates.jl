## STILL IN ALPHA - DOES NOT RUN ON JULIAHUB YET ##

using MLDatasets
using DeepPumas
using SimpleChains
using SimpleChains: relu
using Images
using CairoMakie
using CairoMakie: Axis
Makie.set_theme!(DeepPumas.plottheme_dark())

## Load the MNIST annotated image data set
_xtrain, ytrain = MLDatasets.MNIST(; split=:train)[:]
_xtest, ytest = MLDatasets.MNIST(; split=:test)[:]

## Reshape of suit SimpleChains. Rotate x-y to display the images right-side-up.
xtrain = permutedims(reshape(_xtrain, 28, 28, 1, :), (2, 1, 3, 4))
xtest = permutedims(reshape(_xtest, 28, 28, 1, :), (2, 1, 3, 4))

## Have a peek at a random image from the data set
Gray.(xtrain[:, :, 1, rand(1:end)])

############################################################################################
## Generate synthetic data based on the MNIST annotations.
############################################################################################
_trainpop = [Subject(; id=i, covariates=(; n=ytrain[i], img=xtrain[:, :, :, i])) for i in eachindex(ytrain)]
_testpop = [Subject(; id=i, covariates=(; n=ytest[i], img=xtest[:, :, :, i])) for i in eachindex(ytest)]

datamodel = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0.0)
  end
  @covariates n
  @pre begin
    α = n
    r = 3 * (0.5 + (n + 1) / 10)
  end
  @init y = 5.0
  @dynamics y' = r * (α - y)
  @derived begin
    Y = @. Normal(y, σ)
  end
end

# Mimic that data is not sampled at perfect time intervals and that all patients don't have
# the same number of samples
sample_times() = vcat(0, cumsum(rand(Gamma(7, 0.2 / (7 - 1)), rand(10:20))))

p_truth = (; σ=0.5)
trainpop = [
  Subject(simobs(datamodel, subj, p_truth; obstimes=sample_times())) for subj in _trainpop
]
testpop = [
  Subject(simobs(datamodel, subj, p_truth; obstimes=sample_times())) for subj in _testpop
]

plotgrid(trainpop[1:12])

trainpop[1].covariates(0).n
Gray.(trainpop[1].covariates(0).img[:, :, 1])

############################################################################################
## Fit an NLME model where η slurps heterogeniety
############################################################################################
ηmodel = @model begin
  @param begin
    σ ∈ RealDomain(; lower=0.0)
    tvr ∈ RealDomain(; lower=0.0)
    tvα ∈ RealDomain(; lower=0.0)
    Ω ∈ PSDDomain(2)
  end
  @random begin
    η ~ MvNormal(Ω)
  end
  @pre begin
    α = tvα + η[1]
    r = tvr * exp(η[2])
  end
  @init y = 5.0
  @dynamics y' = r * (α - y)
  @derived Y ~ @. Normal(y, σ)
end

@time fpmη = fit(ηmodel, trainpop, init_params(ηmodel), FOCE())

predη = predict(ηmodel, testpop[1:24], coef(fpmη));
plotgrid(predη)

############################################################################################
## Train the CNN model towards our target η posterior
############################################################################################

target = preprocess(fpmη; covs=(:img,), standardize=false)

## Define a convolutional neural network - a variant of the 'LeNet-5' model.
lenet = SimpleChain(
  (static(28), static(28), static(1)),
  Conv(relu, (5, 5), 6),
  MaxPool(2, 2),
  Conv(relu, (5, 5), 16),
  MaxPool(2, 2),
  Flatten(3),
  TurboDense(relu, 120),
  TurboDense(relu, 84),
  TurboDense(identity, numoutputs(target)),
)

#=
Here, we're changing the output from being a classifier of unordered data to being a
regressor of ordered data. This comes at the cost of a little predictive accuracy but it
often makes sense in a pharmacometric scenario where we typically expect patients to be on a
continum. It also ensures that when an image is very unclear then it's better to fall back
towards the mean prediction rather than just picking whether the scribble should be
interpreted as a 4 or a 9. 
=#

## Fit using SimpleChains.jl machinery
lenetloss = SimpleChains.add_loss(lenet, AbsoluteLoss(target.y))
G = SimpleChains.alloc_threaded_grad(lenetloss)
p = SimpleChains.init_params(lenet)
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 100);

############################################################################################
## Embed the fitted CNN in an NLME model - replacing the effect that the η had before
############################################################################################

deep_model = let _p = Float64.(p), _lenet = lenet, ytrsf = target.ytrsf
  @model begin
    @param begin
      NN ∈ NeuralDomain(_lenet, nothing, _p)
      tvr ∈ RealDomain(; lower=0.0)
      tvα ∈ RealDomain(; lower=0.0)
      σ ∈ RealDomain(; lower=0.0, init=5e-1)
    end
    @covariates img
    @pre begin
      nn_pred := only(ytrsf(NN(img))).η
      α = tvα * exp(nn_pred[1])
      r = tvr * exp(nn_pred[2])
    end
    @init y = 5.0
    @dynamics y' = r * (α - y)
    @derived begin
      Y = @. Normal(y, σ)
    end
  end
end

param = (;
  Base.structdiff(coef(fpmη), (; Ω=nothing))...,
  NN=init_params(deep_model).NN
)

@time pred = predict(deep_model, testpop, param; obstimes=0:0.05:4);
plotgrid(pred[1:25]; ipred=false)

res = map(testpop) do s
  icoef(deep_model, s, param)().α - s.covariates().n
end
mean(x -> abs(x) < 0.5, res)

## View the digits that were used to predict the plotted subjects.
colorview(Gray, mapreduce(vcat, testpop[1:25]) do subj
  subj.covariates().img[:, :, 1]
end)

############################################################################################
## We can go even further and fit everything in concert
############################################################################################
@time fpm_deep = fit(
  deep_model,
  trainpop,
  param,
  NaivePooled();
  # Including the NN in the fit makes things slow but it works.
  constantcoef=(; NN=init_params(deep_model).NN)
);

pred2 = predict(deep_model, testpop, coef(fpm_deep); obstimes=0:0.05:4);
plotgrid(pred2[1:25]; ipred=false)
