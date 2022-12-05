##################################################################
### Instructions:
###     1) Open JuliaHub extension
###     2) Wait until connected
###     3) Click "Use current file" with this file open
###     4) Pick 16 vCPU and 8 GB of memory per vCPU and one Julia
###        process for each *node*
###     5) Write 19 in the number of nodes field because we have
###        18 scenarios + 1 main process
###     6) Limit 2 hours as default
###     7) Pick the Pumas-2.2.1 system image in the Image options
###        (or newer if you have it)
###     8) Click start job
###     9) Navigate to  https://pumasai.juliahub.com/ui/Run
###        to find the jobs and ultimately the results.
###     10) Download the results when the job has finished
##################################################################

using Distributed
using Serialization
using Pumas

##################################################################
### Generate a file name for results
##################################################################
result_filename = tempname() * ".jls"
@info "Created results file name $(result_filename)"
ENV["RESULTS_FILE"] = result_filename

##################################################################
### Model definition
##################################################################
@info "Defining model"
model = @model begin
  @param begin
    θ_CL_parent ∈ RealDomain(lower = 0)
    θ_Q_parent ∈ RealDomain(lower = 0)
    θ_Vc_parent ∈ RealDomain(lower = 0)
    θ_Vp_parent ∈ RealDomain(lower = 0)
    θ_CLfm ∈ RealDomain(lower = 0)
    θ_CL_metab ∈ RealDomain(lower = 0)
    θ_Vc_metab ∈ RealDomain(lower = 0)

    omega ∈ RealDomain(lower = 0.1)

    σ_parent ∈ RealDomain(lower = 0.00001)
    σ_metab ∈ RealDomain(lower = 0.00001)
  end

  @random begin
    eta ~ Normal(0, omega)
  end
  @pre begin
    CL = θ_CL_parent * exp(eta)
    Q = θ_Q_parent
    Vc = θ_Vc_parent
    Vp = θ_Vp_parent
    CLfm = θ_CLfm
    CLm = θ_CL_metab
    Vm = θ_Vc_metab
  end

  @dynamics Central1Periph1Meta1

  @derived begin
    C_parent := @. Central / Vc
    y_parent ~ @. Normal(C_parent, C_parent * σ_parent)
    C_metab := @. Metabolite / Vm
    y_metab ~ @. Normal(C_metab, C_metab * σ_metab)
  end
end

##################################################################
### Stochastic simulation and estimation (sse) function
##################################################################
@everywhere function simpop(scenario, param)
    t, npop, amt = scenario

    dr = DosageRegimen(amt, duration = 1)
    skeleton_pop = [Subject(id = i, events = dr) for i = 1:npop]

    pop = Subject.(simobs(model, skeleton_pop, param; obstimes = t))
end
@everywhere function sse(model, param, scenario, nsim)
    @info "Fitting $scenario"
    # Create DataFrames to hold output
    fits = []
    for i = 1:nsim
        # Simulate based on scenario characteristics
        pop_est = simpop(scenario, param)

        # Refit using evaluation model
        try
            new_res = fit(model, pop_est, param, Pumas.FOCE(); optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
            push!(fits, new_res)
        catch

        end
    end
    (;scenario, param, fits)
end

##################################################################
### Data generating parameters / "true parameters"
##################################################################
@info "Defining parameters"
@everywhere param = (
  θ_CL_parent = 3.0,
  θ_Q_parent = 2.0,
  θ_Vc_parent = 14.0,
  θ_Vp_parent = 6.0,
  θ_CLfm = 1.0,
  θ_CL_metab = 6.5,
  θ_Vc_metab = 15.0,
  omega = 0.2,
  σ_parent = sqrt(0.1),
  σ_metab = sqrt(0.1),
)

##################################################################
### Scenario specification
##################################################################
@info "Defining scenarios"
@everywhere Tsim = [[0.5,1.0,1.5,], [0.5,1.0,1.5,2.0,2.5,3.0,]]
@everywhere Npop = [50, 70, 80]
@everywhere Amt = [200, 300, 500]
@everywhere scenarios = [(;t, npop, amt) for t in Tsim, npop in Npop, amt in Amt]

@everywhere nrep = 200
@info "A total of $(length(scenarios)) scenarios will be simulated and re-estimated $(nrep) times."

results = pmap(i->sse(model, param, scenarios[i], nrep), 1:length(scenarios))

@info "Serialize"
open(result_filename, "w") do io
    serialize(io, results)
end
