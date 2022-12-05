using Pumas
using PharmaDatasets

pkdata = dataset("iv_sd_3")
pop = read_pumas(pkdata)

iv1cmt = @emmodel begin

    @random begin
        CL ~ 1 | LogNormal
        Vc ~ 1 | LogNormal
    end

    @dynamics Central1

    @post begin
        cp = 1000 * Central / Vc
    end

    @error begin
        dv ~ ProportionalNormal(cp)
    end
end

params = (CL = 1.0, Vc = 10.0)

iv1cmt_fit = fit(iv1cmt, pop, params, Pumas.SAEM())
iv1cmt_fit_laplace = fit(iv1cmt, pop, coef(iv1cmt_fit), LaplaceI())

iv1cmt_fit_foce = fit(convert(PumasModel, iv1cmt), pop, Pumas.em_to_m_params(iv1cmt, coef(iv1cmt_fit)), FOCE())

# The workflow is the same as any Pumas model
# Confidence Intervals
infer(iv1cmt_fit)
# Inspect
inspect(iv1cmt_fit)

# To get the MCMC chains from each subject:
icoef(iv1cmt_fit)
