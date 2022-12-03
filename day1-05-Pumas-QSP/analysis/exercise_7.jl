include("exercise_6.jl")
# Exercise 7
using GlobalSensitivity

sens = SensitivityProblem(trial_1, QSPmodel, [k1 =>(1., 2.), k3 => (5., 10.), k2 => (200, 300)])
gsa_res = gsa(sens, 
    Morris(total_num_trajectory = 5000, num_trajectory = 500), 
    batch = false, 
    samples = 10000);
# gsa_res2 = gsa(sens, Sobol(), batch = false, samples = 1000)
# gsa_res3 = gsa(sens, eFAST(), batch = false, samples = 1000)

# How is State y4 effected by changes in k1 and k2?
p1 = scatter(gsa_res.means[4,:], gsa_res.variances[4,:], series_annotations = ["k1", "k2", "k3"], label = "", grid = "off", xlabel = "Mean", ylabel = "Variance", title = "State y4")
savefig("visualization/exercise_7_y4.pdf")
# How is State y7 effected by changes in k1 and k2?
p2 = scatter(gsa_res.means[7,:], gsa_res.variances[7,:], series_annotations = ["k1", "k2", "k3"], label = "", grid = "off", xlabel = "Mean", ylabel = "Variance", title = "State y7")
savefig("visualization/exercise_7_y7.pdf")
