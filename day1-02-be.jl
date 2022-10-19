# Load Libraries
using Bioequivalence
using PharmaDatasets
using DataFramesMeta

# Read Data into DataFrame
pkdata = dataset("bioequivalence/2S2P/PJ2017_3_1")

# BE with Cmax
output_cmax =
    pumas_be(pkdata; endpoint = :Cmax, id = :id, sequence = :sequence, period = :period)

# BE with AUC
output_auc =
    pumas_be(pkdata; endpoint = :AUC, id = :id, sequence = :sequence, period = :period)

# Fed vs. NonFed Example
unique(pkdata.sequence)
@rtransform! pkdata :sequence = sequence == "RT" ? "NF" : "FN"

# BE with Cmax specifying reference as Non-Fed
cmax_output_nonfed = pumas_be(
    pkdata;
    endpoint = :Cmax,
    reference = 'N',
    id = :id,
    sequence = :sequence,
    period = :period,
)


# How to look at individual outputs:
# comparing formulations (R vs T) 
output_auc.data_stats.formulation
output_cmax.data_stats.formulation

# comparing sequences (RT vs TR)
output_auc.data_stats.sequence

# comparing periods (1 vs 2)
output_auc.data_stats.period

# output statistical model results 
output_auc.model

# perform Wald test - assesses whether the model parameters are jointly statistically significant from zero
output_auc.model_stats.Wald

# compare least squares geometric means for the formulation 
output_auc.model_stats.lsmeans

# outputs dataframe of result 
output_auc.result