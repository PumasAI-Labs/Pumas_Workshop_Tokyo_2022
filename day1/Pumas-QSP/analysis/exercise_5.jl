include("exercise_4.jl")

# Exercise 5:
using Distributions, StatsPlots
_vp = CSV.read("visualization/VP_data_1000.csv", DataFrame)
vp = import_vpop(_vp, prob)

# We are having a closer look at: Trial 3
plt3 = plot(vp, trial_3;  ylabel = "State", grid = "off", 
        xlabel = "Time", legend = :outerright, show_data = true, title = "Trial 3")

# Last timepoint 
sim = solve_ensemble(vp, trial_3)
data4 = [sim[i].u[end][4] for i in 1:length(vp)] # y4
data6 = [sim[i].u[end][6] for i in 1:length(vp)] # y6
col_1 = "#1098f2"
col_2 = "#e06a43"
hist = histogram(data6, orientation=:h, label = "y6", nbins = 20, color = col_1,  ylabel = "", title = "Last time point", xlabel = "Patient count", yticks = ([], []), ylim = (0,0.0135), grid = "off", legend = :bottom)
        histogram!(data4, orientation=:h, label = "y4", nbins = 20, color = col_2)
plt3 = plot(vp, trial_3;  states = [y4, y6], ylabel = "State", grid = "off", label = "",
        xlabel = "Time", legend = :outerright, show_data = true, title = "Time course")
plot(plt3, hist)

# Reference distributions 
dist4 = TransformedBeta(Beta = Beta(2, 2), lb = 0.011, ub = 0.013) 
dist6 = TransformedBeta(Beta = Beta(2, 5), lb = 0.01, ub = 0.015) 
ref = [
    y4 => (dist = dist4, t = 10.0),
    y6 => (dist = dist6, t = 10.0)
]
plt_dist = plot(dist4, permute = (:x, :y), label = "y4", color = col_2, grid = "off",  title = "Reference", xlim = (0,0.013), legend = :bottomright)
            plot!(dist6, permute = (:x, :y), label = "y6", color = col_1);
plot(plt3, hist, plt_dist, layout = (1,3)) 

# Run subsampling 
subsample_population_size = 100
alg = DDS(reference = ref, n = subsample_population_size, nbins = 20)
vp_sub = subsample(alg, vp, trial_3)

# Visualize subsapling results
plt3_sub = plot(vp_sub, trial_3;   states = [y4, y6], ylabel = "State", grid = "off", xlabel = "Time", label = "", show_data = true, title = "Trial 3")
sim_sub = solve_ensemble(vp_sub, trial_3)
data4_sub = [sim_sub[i].u[end][4] for i in 1:length(vp_sub)] # y4
data6_sub = [sim_sub[i].u[end][6] for i in 1:length(vp_sub)] # y6
hist_sub = histogram(data6_sub, orientation=:h, label = "y6", color = col_1, ylabel = "State", title = "Last time point", xlabel = "Patient count", ylim = (0,0.013), grid = "off", legend = :bottomright)
        histogram!(data4_sub, orientation=:h, label = "y4", color = col_2)
plot(plt3, hist, plt_dist, plt3_sub, hist_sub, layout = (1, 5), size = (1500, 1000))
savefig("visualization/exercise_5.pdf")

plt3_16 = plot(vp, trial_3;   states = [y6], ylabel = "State", grid = "off", xlabel = "Time", label = "", show_data = true, title = "Time course")
plt3_sub_16 = plot(vp_sub, trial_3;   states = [y6], ylabel = "State", grid = "off", xlabel = "Time", label = "", show_data = true, title = "Time course")
hist6 = histogram(data6, orientation=:h, label = "y6 - 1000 patients", ylabel = "", title = "Last time point", yticks = ([], []), xlabel = "Patient count", ylim = (0,0.013), grid = "off", legend = :bottomright)
hist_sub6 = histogram(data6_sub, orientation=:h, label = "y6 - 100 patients",  ylabel = " ", title = "Last time point", yticks = ([], []), xlabel = "Patient count", ylim = (0,0.013), grid = "off", legend = :bottomright)
plot(plt3_16, hist6, plt3_sub_16, hist_sub6)


plt3_14 = plot(vp, trial_3;   states = [y4], ylabel = "State", grid = "off", xlabel = "Time", label = "", show_data = true, color = col_2,  title = "Time course", ylim = (0,0.013))
plt3_sub_14 = plot(vp_sub, trial_3;   states = [y4], ylabel = "State", grid = "off", xlabel = "Time", label = "", show_data = true, color = col_2, title = "Time course", ylim = (0,0.013))
hist4 = histogram(data4, orientation=:h, label = "y4 - 1000 patients", ylabel = "", title = "Last time point", yticks = ([], []), xlabel = "Patient count", color = col_2, ylim = (0,0.013), grid = "off", legend = :bottomright)
hist_sub4 = histogram(data4_sub, orientation=:h, label = "y4 - 100 patients",  ylabel = "", title = "Last time point", yticks = ([], []), xlabel = "Patient count", color = col_2, ylim = (0,0.013), grid = "off", legend = :bottomright)
plot(plt3_14, hist4, plt3_sub_14, hist_sub4)


