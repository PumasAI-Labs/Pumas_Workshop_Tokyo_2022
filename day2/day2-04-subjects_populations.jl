

model = @model begin
    @param begin
        tvcl ∈ RealDomain(; lower = 0)
        tvvc ∈ RealDomain(; lower = 0)
        tvka ∈ RealDomain(; lower = 0)
        Ω ∈ PSDDomain(3)
        σ_prop ∈ RealDomain(; lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates WT

    @pre begin
        CL = tvcl * (WT / 70)^0.75 * exp(η[1])
        Vc = tvvc * (WT / 70) * exp(η[2])
        Ka = tvka * exp(η[3])
    end

    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Central * CL / Vc
    end

    @derived begin
        conc = @. Central / Vc
        dv ~ @. Normal(conc, conc * σ_prop)
    end

end

# A population is just a collection of subjects. 
# So let us create a subject First

# empty subject
s1 = Subject()
# add an ID
s1 = Subject(id = "Tokyo")
# add dosage regimen
s1 = Subject(id = "Tokyo", events = DosageRegimen(750))
#look at events generated            
DataFrame(s1.events)
# add some covariates
s1 = Subject(id = "Tokyo", events = DosageRegimen(750), covariates = (WT = 70,))
#visualize the subject as a dataframe
DataFrame(s1)
# add sampling times
s1 = Subject(
    id = "Tokyo",
    events = DosageRegimen(750),
    covariates = (WT = 70,),
    time = [0.1, 0.5, 1, 2, 4, 6, 8, 12, 24],
)
#visualize the subject as a dataframe
DataFrame(s1)
# add observations
s1 = Subject(
    id = "Tokyo",
    events = DosageRegimen(750),
    covariates = (WT = 70,),
    time = [0.1, 0.5, 1, 2, 4, 6, 8, 12, 24],
    observations = (dv = nothing,),
)
#visualize the subject as a dataframe
DataFrame(s1)

## simulate a single subject
fixeffs =
    (; tvcl = 0.4, tvvc = 20, tvka = 1.1, Ω = Diagonal([0.04, 0.04, 0.04]), σ_prop = 0.2)

#
sims10 = simobs(model, s1, fixeffs; obstimes = 0:0.1:120)
sim_plot(sims10, observations = [:conc])

# create a second subject
s2 = Subject(
    id = "Yokohama",
    events = DosageRegimen(750),
    covariates = (WT = 100,),
    time = [0.1, 0.5, 1, 2, 4, 6, 8, 12, 24],
    observations = (dv = nothing,),
)

# Create a population of two subjects
s1s2 = [s1, s2]
sims11 = simobs(model, s1s2, fixeffs; obstimes = 0:0.1:120)
sim_plot(sims11, observations = [:conc])
sim_plot(sims11, observations = [:conc], separate = true)

# Create a population of many subjecs at once

manysubjs = map(
    subj -> Subject(
        id = subj,
        events = DosageRegimen(750),
        covariates = (WT = rand(55:90),),
        time = [0.1, 0.5, 1, 2, 4, 6, 8, 12, 24],
        observations = (dv = nothing,),
    ),
    1:10,
)
sims12 = simobs(model, manysubjs, fixeffs; obstimes = 0:0.1:120)
sim_plot(sims12, observations = [:conc])
sim_plot(sims12, observations = [:conc], 
            separate = true,
            facet = (combinelabels = true,))
