# Data generation for CSV files
using ModelingToolkit, DifferentialEquations, DataFrames, CSV, Plots
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

prob1 = ODEProblem(QSPmodel, [], (0.0, 100000.0), [k6 => 20.0])
prob1 = SteadyStateProblem(prob1)
sol1 = solve(prob1, DynamicSS(QNDF()))
data1 = DataFrame(["y1(t)", "y2(t)", "y3(t)", "y4(t)", "y5(t)", "y6(t)", "y7(t)", "y8(t)"] .=> sol1)
CSV.write("data/trial_1.csv", data1)
data1_csv = CSV.read("data/trial_1.csv", DataFrame)

prob2 = ODEProblem(QSPmodel, sol1.u, (0.0, 20.0), [k3 => 4.0, k4 => 0.4, k6 => 8.0])
sol2 = solve(prob2, Rosenbrock23(), abstol=1e-9, reltol=1e-8)
data2 = DataFrame(sol2)
CSV.write("data/trial_2.csv", data2)

prob3 = ODEProblem(QSPmodel, sol1.u, (0.0, 10.0), [k3 => 15.0, k4 => 0.6, k6 => 2.0])
sol3 = solve(prob3, Rosenbrock23(), abstol=1e-9, reltol=1e-8, saveat = Array(0:10))
data3 = DataFrame(sol3)
CSV.write("data/trial_3.csv", data3)

plot(sol1)
plot(sol2)
plot(sol3)





