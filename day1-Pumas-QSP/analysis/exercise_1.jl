# Exercise 1.2
using DataFrames, CSV, Plots
# cd("day1-05-Pumas-QSP")

path_to_data_1 = "data/trial_1.csv"
path_to_data_2 = "data/trial_2.csv"
path_to_data_3 = "data/trial_3.csv"

data_1 = CSV.read(path_to_data_1, DataFrame)
data_2 = CSV.read(path_to_data_2, DataFrame) 
data_3 = CSV.read(path_to_data_3, DataFrame)

plt1 = scatter( Matrix(data_1), xlim = (0, 2), label = "", xlabel = "Steady State", ylabel = "States", legend = :outerright, title = "Trial 1", grid = "off", xticks = ([1],[""]))
savefig("visualization/exercise_1_Trial_1.pdf")
plt2 = scatter(data_2[!,1], Matrix(data_2[:,2:end]), label = "", xlabel = "Time", ylabel = "States", legend = :outerright, title = "Trial 2")
    plot!(data_2[!,1], Matrix(data_2[:,2:end]), label = "", color = "grey", xlabel = "Time", ylabel = "States", legend = :outerright, title = "Trial 2", grid = "off")
savefig("visualization/exercise_1_Trial_2.pdf")
plt3 = scatter(data_3[!,1], Matrix(data_3[:,2:end]), legend = :outerright, xlabel = "Time", ylabel = "States", title = "Trial 3")
    plot!(data_3[!,1], Matrix(data_3[:,2:end]), color = "grey", label = "", legend = :outerright, xlabel = "Time", ylabel = "States", title = "Trial 3", grid = "off")
savefig("visualization/exercise_1_Trial_3.pdf")
plts = plot(plt1, plt2, plt3)
savefig("visualization/exercise_1.pdf")