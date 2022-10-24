# load libraries
using CSV
using DataFramesMeta
using Dates
using NCA
using NCAUtilities
using NCA.Unitful
using PumasUtilities
using CairoMakie
using PharmaDatasets  #this package is used for the example dataset


## Load Data
pk_data = PharmaDatasets.dataset("iv_sd_1") #load dataset from PharmaDatasets 

first(pk_data, 6) # display first 6 rows

## Specify the units used in the analysis
timeu = u"hr"
concu = u"mg/L"
amtu = u"mg"

################################# Single dose NCA#################################################

## Map DataFrame to NCA Population
pop_bolus_sd = read_nca(
    pk_data;
    id = :id,
    time = :time,
    observations = :conc,
    amt = :amt,
    route = :route,
    timeu = true,
    amtu = true,
    concu = true,
    llq = 0.001,# from a bioassay
    group = [:dose],
)
# note that if your dataset has the columns named in the style of read_nca you do not need to mapping


# Preview data  
## Visualize individual concentration-time curve - linear scale 
obsvstimes = observations_vs_time(pop_bolus_sd[7])

## Visualize individual concentration-time curve - semi-log scale
obsvstimes = observations_vs_time(pop_bolus_sd[7]; axis = (yscale = log,))

## Visualize the mean concentration-time curve of population 
summary_observations_vs_time(
    pop_bolus_sd;
    axis = (xlabel = "Time (hour)", ylabel = "Drug Concentration (mg/L)"),
)

## Run an annotated NCA to be used for a report
nca_bolus_sd_report = run_nca(
    pop_bolus_sd;
    sigdigits = 3,
    studyid = "STUDY-001",
    studytitle = "Drug Trial: IV Bolus", # required
    author = [("Author 1")], # required
    sponsor = "PumasAI",
    date = Dates.now(),
    conclabel = "Drug Concentration (mg/L)",
    timelabel = "Time (hr)",
    versionnumber = v"0.1",
)

## Summarize results of interest in a clean report
param_summary_bolus_sd = summarize(
    nca_bolus_sd_report.reportdf;
    stratify_by = [:dose],
    parameters = [:half_life, :tmax, :cmax, :auclast, :vz_obs, :cl_obs, :aucinf_obs],
)

## Generate NCA report 
report(nca_bolus_sd_report, param_summary_bolus_sd)

## If you want to just generate an individual NCA parameter
vz = NCA.vz(pop_bolus_sd)  # Volume of Distribution
cl = NCA.cl(pop_bolus_sd)  # Clearance
lambdaz = NCA.lambdaz(pop_bolus_sd; threshold = 3)  # Terminal Elimination Rate Constant, threshold=3 specifies the max no. of time point used for calculation
lambdaz_1 = NCA.lambdaz(pop_bolus_sd; slopetimes = [8, 12, 16]) # slopetimes in this case specifies the exact time point you want for the calculation
thalf = NCA.thalf(pop_bolus_sd) # Half-life calculation
cmax_d = NCA.cmax(pop_bolus_sd; normalize = true) # Dose Normalized Cmax
mrt = NCA.mrt(pop_bolus_sd) # Mean residence time
aumc = NCA.aumc(pop_bolus_sd; method = :linlog) # AUMC calculation, using :linlog method
individual_params = innerjoin(
    vz,
    cl,
    lambdaz,
    lambdaz_1,
    thalf,
    cmax_d,
    mrt,
    aumc;
    on = [:id, :dose],
    makeunique = true,
)

## Calculation of AUC at specific time intervals and merge to the final report DataFrame
auc0_12 = NCA.auc(pop_bolus_sd; interval = (0, 12), method = :linuplogdown) #various other methods are :linear, :linlog
auc12_24 = NCA.auc(pop_bolus_sd; interval = (12, 24), method = :linuplogdown)
final = innerjoin(
    nca_bolus_sd_report.reportdf,
    auc0_12,
    auc12_24;
    on = [:id],
    makeunique = true,
)

report(nca_bolus_sd_report, param_summary_bolus_sd)



############################### NCA Multiple Dose First and SS dose ###############################


## Load Data
pk_md_data = dataset("nlme_sample.csv") #load dataset from PharmaDatasets 

## Add ii column to tell us how often dose is given
@rtransform! pk_md_data begin
    :ii = 24
    :addl = :TIME == 0 ? 5 : missing
end

## select first dose and a ss dose example
@rsubset! pk_md_data begin
    :OCC == 1 || :OCC == 7
    :ID != 28
end


## Tell NCA that this measurement is at steady state ...  you can also run with just the occ 7 data and this ss flag 
@rtransform! pk_md_data :ss = :TIME > 24 ? 1 : 0

## Specify the units used in the analysis
timeu = u"hr"
concu = u"mg/L"
amtu = u"mg"


## Map DataFrame to NCA Population
pop_inf_md = read_nca(
    pk_md_data;
    id = :ID,
    time = :TIME,
    observations = :DV,
    amt = :AMT,
    route = :ROUTE,
    ii = :ii,             # this will allow us to get our Tau results   
    addl = :addl,           # how many doses were given in beween dosing rows
    ss = :ss,             # a flag to label when we are at ss
    duration = :DURATION,       # Because we are using an infusion
    timeu = true,
    amtu = true,
    concu = true,
    llq = 0.001,
    group = [:GROUP, :OCC],
) # we want to be able to compare the differnt doses and first dose vs ss.

nca_inf_md_report = run_nca(
    pop_inf_md;
    sigdigits = 3,
    studyid = "STUDY-002",
    studytitle = "Drug Trial: Infusion Multiple Dose", # required
    author = [("Author 1")], # required
    sponsor = "PumasAI",
    date = Dates.now(),
    conclabel = "Drug Concentration (mg/L)",
    timelabel = "Time (hr)",
    versionnumber = v"0.1",
)

param_summary_inf_md = summarize(
    nca_inf_md_report.reportdf,
    stratify_by = [:GROUP, :OCC],
    parameters = [:half_life, :tmax, :cmax, :auclast, :vz_obs, :cl_obs, :auc_tau_obs],
)
report(nca_inf_md_report, param_summary_inf_md)
