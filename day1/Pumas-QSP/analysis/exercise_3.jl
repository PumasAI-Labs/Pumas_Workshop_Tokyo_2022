include("exercise_1.jl")
include("exercise_2.jl")
# Exercise 3.1
using PumasQSP, DifferentialEquations
# SteadyStateTrial - Normalization/Adaption
trial_1 = SteadyStateTrial(collect(data_1[1, :]), QSPmodel;
                alg=DynamicSS(QNDF()),
                abstol=1e-9, reltol=1e-8,
                params=[k6 => 20.0]
                )
# Trial - Long term effect
trial_2 = Trial(data_2, QSPmodel;
                tspan=(0.0, 20.0),
                alg=Rosenbrock23(),
                abstol=1e-9, reltol=1e-8,
                forward_u0=true,
                params=[k3 => 4.0, k4 => 0.4, k6 => 8.0]
                )
# Trial - Short term effect
trial_3 = Trial(data_3, QSPmodel;
                tspan=(0.0, 10.0),
                alg=Rosenbrock23(),
                abstol=1e-9, reltol=1e-8,
                forward_u0=true,
                params=[k3 => 15.0, k4 => 0.6, k6 => 2.0]
                )

# Exercise 3.2
trials = SteadyStateTrials(trial_1, trial_2, trial_3)