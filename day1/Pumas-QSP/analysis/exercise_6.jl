include("exercise_5.jl")
# Exercise 6
v_trial_1 = virtual_trial(trial_3, QSPmodel; params = [k3 => 1.0, k4 => 1.7])
plt = plot(vp, v_trial_1, grid = "off", ylabel = "State", xlabel = "Time", legend = :outerright)
savefig("visualization/exercise_6.pdf")