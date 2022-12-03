# cd("PumasQSPWorkshop")
include("exercise_7.jl")
# Exercise 8
using Test
# Test for exercise 1
@test size(data_1) == (1, 8)
@test size(data_2) == (124, 9)
@test size(data_3) == (11, 9)
#Test for exercise 2
@test sol.t[end] == 10.
#Test for exercise 3
@test length(Array(solve_trial(trial_1, vp[1], prob))) == size(data_1)[2]
@test size(Array(solve_trial(trial_2, vp[1], prob))) == (size(data_2)[2]-1, size(data_2)[1])
@test size(Array(solve_trial(trial_3, vp[1], prob))) == (size(data_3)[2]-1, size(data_3)[1])
# Test for exercise 4
# @test length(vp) == population_size
@test length(vp[1]) == 3
cost = objective(prob, StochGlobalOpt(maxiters = 750))
@test all(cost.(vp) .<1e-3)
# Test for exercise 5
@test size(Array(solve_trial(v_trial_1, vp[1], prob))) == (8, 11)
@test solve_trial(v_trial_1, vp[1], prob).t[end] == 10.
