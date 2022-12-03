# Exercise 9

# Same as in Exercise 1, 2, 3, 4
using DataFrames, CSV, ModelingToolkit, DifferentialEquations

# Change 1: use Distributed pkg
using Distributed

# Change 2: use @everywhere macro
@everywhere using PumasQSP

# Change 3: Path description 
datadir = joinpath(@__DIR__, "../data/")
path_to_data_1 = joinpath(datadir, "trial_1.csv")
path_to_data_2 = joinpath(datadir, "trial_2.csv")
path_to_data_3 = joinpath(datadir, "trial_3.csv")

# Same as in Exercise 1, 2, 3, 4
data_1 = CSV.read(path_to_data_1, DataFrame)
data_2 = CSV.read(path_to_data_2, DataFrame) 
data_3 = CSV.read(path_to_data_3, DataFrame)
@variables begin
    t 
    y1(t) = 1.0
    y2(t) = 0.0
    y3(t) = 0.0
    y4(t) = 0.0
    y5(t) = 0.0
    y6(t) = 0.0
    y7(t) = 0.0
    y8(t) = 0.0057
end
@parameters begin
    k1 = 1.71
    k2 = 280.0
    k3 = 8.32
    k4 = 0.69
    k5 = 0.43 
    k6 = 1.81
end
D = Differential(t)
eqs = [
    D(y1) ~ (-k1*y1 + k5*y2 + k3*y3 + 0.0007),
    D(y2) ~ (k1*y1 - 8.75*y2),
    D(y3) ~ (-10.03*y3 + k5*y4 + 0.035*y5),
    D(y4) ~ (k3*y2 + k1*y3 - 1.12*y4),
    D(y5) ~ (-1.745*y5 + k5*y6 + k5*y7),
    D(y6) ~ (-k2*y6*y8 + k4*y4 + k1*y5 - k5*y6 + k4*y7),
    D(y7) ~ (k2*y6*y8 - k6*y7),
    D(y8) ~ (-k2*y6*y8 + k6*y7)
]
@named QSPmodel = ODESystem(eqs)
trial_1 = SteadyStateTrial(collect(data_1[1, :]), QSPmodel;
                alg = DynamicSS(QNDF()),
                abstol = 1e-9, reltol = 1e-8,
                params = [k6 => 20.0]
                )
trial_2 = Trial(data_2, QSPmodel;
                tspan = (0.0, 20.0),
                alg = Rosenbrock23(),
                abstol = 1e-9, reltol = 1e-8,
                forward_u0 = true,
                params = [k3 => 4.0, k4 => 0.4, k6 => 8.0]
                )
trial_3 = Trial(data_3, QSPmodel;
                tspan = (0.0, 10.0),
                alg = Rosenbrock23(),
                abstol = 1e-9, reltol = 1e-8,
                forward_u0 = true,
                params = [k3 => 15.0, k4 => 0.6, k6 => 2.0]
                )
trials = SteadyStateTrials(trial_1, trial_2, trial_3)
prob = InverseProblem(trials, QSPmodel, [k1 => (0. , 5.), k2 => (200., 300.), y1 => (0. , 3.)])
population_size = 10 # 10_000

# Change 4: specify parallel_type
vp = vpop(prob, StochGlobalOpt(maxiters = 750, parallel_type = EnsembleDistributed()); population_size)

# Change 5: save file via environment  
CSV.write("exercise_9_vpop_10.csv", vp)
ENV["RESULTS_FILE"] = joinpath(pwd(), "exercise_9_vpop_10.csv")