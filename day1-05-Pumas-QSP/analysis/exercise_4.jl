include("exercise_3.jl")

# Exercise 4.1
prob = InverseProblem(trials, QSPmodel, [k1 => (0. , 5.), k2 => (200., 300.), y1 => (0. , 3.)])

# Exercise 4.2
population_size = 10
vp = vpop(prob, StochGlobalOpt(maxiters = 750), population_size = population_size)

# Exercise 4.3 
# plt1 = plot(vp, trial_1; grid = "off", ylabel = "State", xlabel = "Time", legend = :outerright, show_data = true)
plt2 = plot(vp, trial_2; grid = "off", 
        ylabel = "State", xlabel = "Time", legend = :outerright, show_data = true, title = "Trial 2")
savefig("visualization/exercise_4_trial_2.pdf")
plt3 = plot(vp, trial_3; grid = "off", ylabel = "State", 
        xlabel = "Time", legend = :outerright, show_data = true, title = "Trial 3")
savefig("visualization/exercise_4_trial_3.pdf")
plot(plt2, plt3)

# Exercise 4.4
vp_df = DataFrame(vp)
CSV.write(joinpath("visualization/VP_data.csv"), vp_df)


# Things to highlight:
# Time bar
# Session Time
# git
# typeof(vp)
# ?VpopResult