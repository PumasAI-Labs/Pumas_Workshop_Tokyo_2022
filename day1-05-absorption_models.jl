using Pumas
using PharmaDatasets
using DataFramesMeta

pkdata = dataset("po_md_2")
pop_oral = read_pumas(pkdata)
pop_dur = read_pumas(@rtransform pkdata :rate = -2)


# First-Order Absorption
foabs = @model begin

    @param begin
        tvcl ∈ RealDomain(lower = 0)
        tvvc ∈ RealDomain(lower = 0)
        tvka ∈ RealDomain(lower = 0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
        Ka = tvka * exp(η[3])
    end

    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL / Vc) * Central
    end

    @derived begin
        cp := @. 1000 * Central / Vc
        dv ~ @. Normal(cp, σ)
    end
end

param_foabs = (tvcl = 5, tvvc = 20, tvka = 1, Ω = Diagonal([0.04, 0.04, 0.04]), σ = 1.0)

fit_foabs = fit(foabs, pop_oral, param_foabs, Pumas.FOCE())

# Zero-Order Absorption
zoabs = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0)
        tvvc ∈ RealDomain(lower = 0)
        tvdur ∈ RealDomain(lower = 0)
        Ω ∈ PDiagDomain(3)
        σ ∈ RealDomain(lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
    end

    @dosecontrol begin
        duration = (Central = tvdur * exp(η[3]),)
    end

    @dynamics begin
        Central' = -(CL / Vc) * Central
    end

    @derived begin
        cp := @. 1000 * Central / Vc
        dv ~ @. Normal(cp, σ)
    end

end

param_zoabs = (tvcl = 5, tvvc = 20, tvdur = 0.3, Ω = Diagonal([0.04, 0.04, 0.04]), σ = 1.0)

fit_zoabs = fit(zoabs, pop_dur, param_zoabs, Pumas.FOCE())

# Two Parallel First-Order Processes
two_parallel_foabs = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0)
        tvvc ∈ RealDomain(lower = 0)
        tvka1 ∈ RealDomain(lower = 0)
        tvka2 ∈ RealDomain(lower = 0)
        tvbio ∈ RealDomain(lower = 0, upper = 1)
        Ω ∈ PDiagDomain(5)
        σ ∈ RealDomain(lower = 0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
        Ka1 = tvka1 * exp(η[3])
        Ka2 = tvka2 * exp(η[4])
    end

    @dosecontrol begin
        bioav = (IR = tvbio * exp(η[5]), SR = (1 - tvbio) * exp(η[5]))
    end

    @dynamics begin
        IR' = -Ka1 * IR # Immediate Release
        SR' = -Ka2 * SR # Sustained Release
        Central' = Ka1 * IR + Ka2 * SR - Central * CL / Vc
    end

    @derived begin
        cp := @. 1000 * Central / Vc
        dv ~ @. Normal(cp, σ)
    end
end

param_two_parallel_foabs = (
    tvcl = 5,
    tvvc = 50,
    tvka1 = 0.8,
    tvka2 = 0.6,
    tvbio = 0.5,
    Ω = Diagonal([0.04, 0.04, 0.36, 0.36, 0.04]),
    σ = 1.0,
)

fit_two_parallel_foabs =
    fit(two_parallel_foabs, pop_oral, param_two_parallel_foabs, Pumas.FOCE())
