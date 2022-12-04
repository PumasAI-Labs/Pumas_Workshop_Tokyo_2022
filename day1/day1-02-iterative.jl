## load libraries
## Core Pumas libraries
using Pumas
using PumasUtilities

## Data handling libraries
using CSV
using DataFramesMeta
using Dates

## Plotting libraries
using CairoMakie
using AlgebraOfGraphics

## Library of default datasets
using PharmaDatasets

## Used to save and call model fits
using Serialization

## load data
pk_md_data_csv = dataset("nlme_sample.csv", String) # load dataset from PharmaDatasets 
pk_data = CSV.read(pk_md_data_csv, DataFrame)

## map population
pop = read_pumas(
    pk_data,
    id = :ID,
    time = :TIME,
    amt = :AMT,
    covariates = [:WT, :AGE, :SEX, :CRCL, :GROUP],
    observations = [:DV],
    cmt = :CMT,
    evid = :EVID,
    rate = :RATE,
)

########################### 1 COMPARTMENT MODEL  ##################                

pk_1cmt = @model begin
    @metadata begin
        desc = "base model: 1comp"
        timeu = u"hr"
    end
    @param begin
        "Clearance (L/hr)"
        tvcl ∈ RealDomain(lower = 0)
        "Volume (L)"
        tvvc ∈ RealDomain(lower = 0)
        """
        - ΩCL
        - ΩVc
        """
        Ω ∈ PDiagDomain(2)
        "Proportional RUV (variance )"
        σ²_prop ∈ RealDomain(lower = 0.0001)
        "Additive RUV (variance) "
        σ²_add ∈ RealDomain(lower = 0.0001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
    end

    @dynamics Central1
    #Central' = -CL/Vc*Central

    @derived begin
        cp := @. 1000 * (Central / Vc)  #cp is suppressed from the output because of  ":="
        """
        Drug Concentration (ng/mL)
        """
        # additive error model 
        #DV ~ @. Normal(CONC, sqrt(σ²_add)) 

        # proportional error model 
        #DV ~ @. Normal(CONC, sqrt(CONC^2*σ²_prop))

        # combination error model 
        DV ~ @. Normal(cp, √(abs(cp)^2 * σ²_prop + σ²_add))
    end
end

params_1cmt_comb =
    (tvvc = 5, tvcl = 0.2, Ω = Diagonal([0.09, 0.09]), σ²_add = 0.01, σ²_prop = 0.01)

##  compute initial log-likelihood (equivalent to maxeval=0)                     
initial_ll = loglikelihood(pk_1cmt, pop, params_1cmt_comb, FOCE())

##  look at the influential individuals
## used to catch data errors, bad initial estimates or model issues                    
pk_influential = findinfluential(pk_1cmt, pop, params_1cmt_comb, FOCE())

# fit the model
pkfit_1cmt_comb = fit(pk_1cmt, pop, params_1cmt_comb, FOCE())

## show model metrics 
metrics_pkfit_1cmt_comb = metrics_table(pkfit_1cmt_comb)
rename!(metrics_pkfit_1cmt_comb, :Value => :comb_1cmp)
## Make on the inference on the estimted parameters
infer_1cmt_comb = infer(pkfit_1cmt_comb) # computes variance-covariance matrix CI calculated as the (1-level)/2 and (1+level)/2
coeftable(infer_1cmt_comb) # returns a table with parameter, se, ci lower and ci upper

## inspect your model diagnostics
inspect_1cmt_comb = inspect(pkfit_1cmt_comb) # get predictions and residuals, ebes, individual parameters 
inspect_df_1cmt_comb = DataFrame(inspect_1cmt_comb)  # convert to a DataFrame

## Post processing plots
goodness_of_fit(inspect_1cmt_comb) # plot will show you 4 basic goodness of fit plots (more day 2)

## Save your fitted model 
# serialize("pkfit_1cmt_comb.jls", pkfit_1cmt_comb)

## Call your saved fitted model 
# pkfit_1cmt_comb = deserialize("pkfit_1cmt_comb.jls")

########################### 2 COMPARTMENT MODEL  ##################                


pk_2cmt = @model begin
    @param begin
        "Clearance (L/hr)"
        tvcl ∈ RealDomain(lower = 0.0001)
        "Volume Central Compartment (L)"
        tvvc ∈ RealDomain(lower = 0.0001)
        "Intercompartmental Clearance (L/h)"
        tvq ∈ RealDomain(lower = 0.0001)
        "Volume Peripheral Compartment (L)"
        tvvp ∈ RealDomain(lower = 0.0001)
        """
          - ΩCL
          - ΩVc
          - ΩQ
          - ΩVp
          """
        Ω ∈ PDiagDomain(2)
        "Additive RUV (variance )"
        σ²_add ∈ RealDomain(lower = 0.0001)
        "Proportional RUV (variance )"
        σ²_prop ∈ RealDomain(lower = 0.0001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        Vc = tvvc * exp(η[2])
        Q = tvq
        Vp = tvvp
    end

    @dynamics Central1Periph1
    # Central'    = -(CL+Q)/Vc*Central + Q/Vp*Peripheral
    # Peripheral' =       Q/Vc*Central - Q/Vp*Peripheral

    @derived begin
        cp := @. Central / Vc
        """
        Drug Concentration (ng/mL)
        """
        DV ~ @. Normal(cp, √(abs(cp)^2 * σ²_prop + σ²_add)) # using variance
    end
end


params_2cmt_comb = (
    tvcl = 0.02,
    tvvc = 5,
    tvq = 0.01,
    tvvp = 10,
    Ω = Diagonal([0.09, 0.09]),
    σ²_add = 0.01,
    σ²_prop = 0.01,
)

## Maximum likelihood estimation
pkfit_2cmt_comb = fit(pk_2cmt, pop, params_2cmt_comb, FOCE())

##  you can quickly check proportional and additive error as well
pkfit_2cmt_add = fit(
    pk_2cmt,
    pop,
    params_2cmt_comb,
    constantcoef = (σ²_prop = 0,), # fixes proportional parameter to a fixed value
    FOCE(),
)

pkfit_2cmt_prop = fit(
    pk_2cmt,
    pop,
    params_2cmt_comb,
    constantcoef = (σ²_add = 0,), # fixes additive parameter to a fixed value
    FOCE(),
)

# Serialize fits 
# serialize("pk_fit_2cmt_comb.jls", pkfit_2cmt_comb)
# pkfit_2cmt_comb = deserialize("pk_fit_2cmt_comb.jls")


## Show Model diagnstic criteria (more day 2)
metrics_pkfit_2cmt_comb = rename((metrics_table(pkfit_2cmt_comb)), :Value => :comb_2cmp) # best model
metrics_pkfit_2cmt_add = rename((metrics_table(pkfit_2cmt_add)), :Value => :add_2cmp)
metrics_pkfit_2cmt_prop = rename((metrics_table(pkfit_2cmt_prop)), :Value => :prop_2cmp)


## Compare the estimates with 1-cmt model
compare_estimates(; pkfit_2cmt_comb, pkfit_1cmt_comb)

## compare model metrics of 1 cmt to 2 cmt (more day 2)
df_compartment_comp = innerjoin(
    metrics_pkfit_2cmt_comb,
    metrics_pkfit_1cmt_comb,
    on = :Metric,
    makeunique = true,
)

## Perform a loglikelihood ratio test
lrt = lrtest(pkfit_1cmt_comb, pkfit_2cmt_comb)
pvalue(lrt)

## Make an inference on the parameter estimates
infer_2cmt_comb = infer(pkfit_2cmt_comb) # computes variance-covariance matrix.. CI  calculated as the (1-level)/2 and (1+level)/2
coeftable(infer_2cmt_comb) #returns a table with parameter, se, ci lower and ci upper

## evaluate the diagnostics
inspect_2cmt_comb = inspect(pkfit_2cmt_comb) # get predictions, residuals, EBEs and individual parameters
inspect_df_2cmt_comb = DataFrame(inspect_2cmt_comb)  # convert to a DataFrame

goodness_of_fit(inspect_2cmt_comb) # plot will show you 4 basic goodness of fit plots (more day 2)

## customized plotting
figure = Figure(; fontsize = 22, resolution = (1400, 800))
wresiduals_vs_time(figure[1, 1], inspect_1cmt_comb, axis = (; title = "1 comp model"))
wresiduals_vs_time(figure[1, 2], inspect_2cmt_comb, axis = (; title = "2 comp model"))
figure

##  Select the two compartment model as our final base model 

#################### COVARIATE MODEL BUILDING #######################################

## some basic plots (more day 2)
covariates_check(pop)
covariates_dist(pop)

empirical_bayes_vs_covariates(inspect_2cmt_comb, categorical = [:SEX, :GROUP])

empirical_bayes_vs_covariates(inspect_2cmt_comb, covariates = [:WT, :AGE, :CRCL])

empirical_bayes_vs_covariates(
    inspect_2cmt_comb,
    covariates = [:SEX, :GROUP],
    categorical = [:SEX, :GROUP],
)

###################### WEIGHT ##########################################


pk_base_wt = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.0001)
        tvvc ∈ RealDomain(lower = 0.0001)
        tvq ∈ RealDomain(lower = 0.0001)
        tvvp ∈ RealDomain(lower = 0.0001)
        Ω ∈ PDiagDomain(2)
        σ²_add ∈ RealDomain(lower = 0.0001)
        σ²_prop ∈ RealDomain(lower = 0.0001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates WT

    @pre begin
        # wtCL = (WT/70)^0.75
        # wtV  = (WT/70)

        CL = tvcl * (WT / 70)^0.75 * exp(η[1])
        Vc = tvvc * (WT / 70) * exp(η[2])
        Q = tvq * (WT / 70)^0.75
        Vp = tvvp * (WT / 70)
    end

    @dynamics Central1Periph1

    @derived begin
        cp := @. Central / Vc
        DV ~ @. Normal(cp, √(abs(cp)^2 * σ²_prop + σ²_add))
    end
end

params_base_wt = (
    tvvc = 5,
    tvcl = 0.02,
    tvq = 0.01,
    tvvp = 10,
    Ω = Diagonal([0.09, 0.09]),
    σ²_add = 0.01,
    σ²_prop = 0.01,
)
#


## Maximum likelihood estimation
pkfit_base_wt = fit(pk_base_wt, pop, params_base_wt, FOCE())

# serialize("pkfit_base_wt.jls", pkfit_base_wt)
# pkfit_base_wt = deserialize("pkfit_base_wt.jls")


## we again could do all of the post model processing - metrics, plots and compare

###################### CRCL ##########################################


pk_base_wt_crcl = @model begin
    @param begin
        "Clearance (L/hr)"
        tvcl ∈ RealDomain(lower = 0.0001)
        "Volume Central Compartment (L)"
        tvvc ∈ RealDomain(lower = 0.0001)
        "Intercompartmental Clearance (L/h)"
        tvq ∈ RealDomain(lower = 0.0001)
        "Volume Peripheral Compartment (L)"
        tvvp ∈ RealDomain(lower = 0.0001)
        θcrcl ∈ RealDomain()
        """
          - ΩCL
          - ΩVc
          - ΩQ
          - ΩVp
          """
        Ω ∈ PDiagDomain(2)
        "Additive RUV (variance )"
        σ²_add ∈ RealDomain(lower = 0.0001)
        "Proportional RUV (variance )"
        σ²_prop ∈ RealDomain(lower = 0.0001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates begin
        "Dose (mg)"
        GROUP
        "Sex"
        SEX
        "Age (years)"
        AGE
        "Weight (kg)"
        WT
        "Creatine Clearance"
        CRCL
    end

    @pre begin
        wtCL = (WT / 70)^0.75    # allometric scaling 
        wtV = (WT / 70)
        crcl_eff = (CRCL / 95)^θcrcl   #posible creatine clearance effect
        CL = tvcl * crcl_eff * wtCL * exp(η[1])
        Vc = tvvc * wtV * exp(η[2])
        Q = tvq * wtCL
        Vp = tvvp * wtV
    end

    @dynamics Central1Periph1

    @derived begin
        cp := @. Central / Vc
        """
        Drug Concentration (ng/mL)
        """
        DV ~ @. Normal(cp, √(abs(cp)^2 * σ²_prop + σ²_add))
    end
end


params_base_wt_crcl = (
    tvvc = 5,
    tvcl = 0.02,
    tvq = 0.01,
    tvvp = 10,
    θcrcl = 0.1,
    Ω = Diagonal([0.09, 0.09]),
    σ²_add = 0.01,
    σ²_prop = 0.01,
)


## Maximum likelihood estimation
pkfit_base_wt_crcl = fit(pk_base_wt_crcl, pop, params_base_wt_crcl, FOCE())


# serialize("pkfit_base_wt_crcl.jls", pkfit_base_wt_crcl)
# pkfit_base_wt_crcl = deserialize("pkfit_base_wt_crcl.jls")
infer_base_wt_crcl = infer(pkfit_base_wt_crcl) # computes variance-covariance matrix.. CI  calculated as the (1-level)/2 and (1+level)/2
coeftable(infer_base_wt_crcl) #returns a table with parameter, se, ci lower and ci upper

inspect_base_wt_crcl = inspect(pkfit_base_wt_crcl) # get predictions and residuals 
inspect_df_base_wt_crclb = DataFrame(inspect_base_wt_crcl)  # convert to a DataFrame

goodness_of_fit(inspect_base_wt_crcl) # plot will show you 4 basic goodness of fit plots (more day 2)

## customized plotting
figure = Figure(; fontsize = 22, resolution = (1000, 800))
empirical_bayes_vs_covariates(
    figure[1, 1],
    inspect_2cmt_comb,
    covariates = :WT, 
)
empirical_bayes_vs_covariates(
    figure[1, 2],
    inspect_base_wt_crcl,
    covariates = :WT,
)
figure


## write a final report
report(
    (; final_model = (pkfit_base_wt_crcl, infer_base_wt_crcl, inspect_base_wt_crcl)),
    categorical = [:SEX, :GROUP],
    date = Dates.now(),
    output = "drug_x_report",
    clean = false,
    title = "Population Pharmacokinetic Analysis for Drug",
    author = "Tokyo",
    version = "v0.1",
    header = "Pumas Report",
    footer = "Tokyo Pumas WS",
)