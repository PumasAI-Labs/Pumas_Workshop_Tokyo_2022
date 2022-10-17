# Don't forget to check: https://tutorials.pumas.ai/html/DataWranglingInJulia/04-read_data.html

using CSV
using DataFramesMeta

##########################
#     I/O CSV Files      #
##########################

df = CSV.read("data/iv_sd_demogs.csv", DataFrame)

# Different delimiters and decimals
# using the keyword arguments `delim` and `decimal`
df_eu = CSV.read("data/iv_sd_demogs_eu.csv", DataFrame; delim=';', decimal=',')

# Custom types for columns
df_custom_types = CSV.read(
    "data/iv_sd_demogs.csv",
    DataFrame;
    types=Dict(:ID => String, :ISMALE => Bool) # using Dict as types
)

df_custom_types = CSV.read(
    "data/iv_sd_demogs.csv",
    DataFrame;
    typemap=Dict(Int64 => String) # using Dict as typemap
)

# Selecting and dropping columns
df_select_names = CSV.read("data/iv_sd_demogs.csv", DataFrame; select=["ID", "AGE"]) # column names as Strings
df_select_names = CSV.read("data/iv_sd_demogs.csv", DataFrame; select=[:ID, :AGE]) # column names as Symbols
df_select_idxs = CSV.read("data/iv_sd_demogs.csv", DataFrame; select=[1, 2]) # column names as indices  
df_select_intervals = CSV.read("data/iv_sd_demogs.csv", DataFrame; select=1:3) # column names as intervals  

df_drop_names = CSV.read("data/iv_sd_demogs.csv", DataFrame; drop=["SCR", "eGFR"]) # column names as Strings
df_drop_names = CSV.read("data/iv_sd_demogs.csv", DataFrame; drop=[:SCR, :eGFR]) # column names as Symbols
df_drop_idx = CSV.read("data/iv_sd_demogs.csv", DataFrame; drop=[4, 6]) # column names as indices
df_drop_intervals = CSV.read("data/iv_sd_demogs.csv", DataFrame; drop=4:6) # column names as intervals

# Missing values
df_missing = CSV.read(
    "data/iv_sd_demogs_missing.csv",
    DataFrame;     # take a look at the first row
    missingstring=["NA", "I don't know prof", "?", "."] # several missing values
    # missingstring = "." # single missing value
)

##########################
#     DataFramesMeta     #
##########################

# Don't forget to check: https://tutorials.pumas.ai/html/DataWranglingInJulia/05-mutating-dfmeta.html

############################################################################################
# dplyr has 50+ functions `mutate`, `mutate_if`, `mutate_at`, `mutate_all`, `rename_with`, #
# `transmute`, `transmute_if`, `transmute_at`, `transmute_all`                             #
############################################################################################

##############################################
# DataFramesMeta has 6 macros and 1 function #
##############################################

## @[r]select[!] and @[r]transform[!]
@select df :ID :AGE

# We can also use `Not()`, `Between()`, RegEx
@select df $(Between(:SCR, :eGFR))
@select df $(Not([:SCR, :eGFR]))
@select df $(r"R$")  # ending with `R`

@select df begin
    :ID
    :eGFR_z = begin
        μ = mean(:eGFR)
        σ = std(:eGFR)
        [(x - μ) / σ for x in :eGFR]
    end
end

@transform df begin
    :eGFR_z = begin
        μ = mean(:eGFR)
        σ = std(:eGFR)
        [(x - μ) / σ for x in :eGFR]
    end
end

@transform df :AGE_log = log(:AGE)  # this errors! Why?!
@transform df :AGE_log = log(sqrt(abs(:AGE))) + 10
@rtransform df :AGE_log = log(:AGE) # this doesn't! Why?!

@rselect df :ID :AGE_log = log(:AGE)

@transform df :AGE_diff_mean = :AGE .- mean(:AGE)

## mutating non-allocating [!] macros

@rtransform! df :AGE_log = log(:AGE)

# dplyr (actually magrittr) has a an "Assignment pipe": `%<>%`
# df %<>%
#     select(col1, col2)

## @[r]subset[!]
@rsubset df :eGFR > 100
@rsubset df :eGFR > 100 :AGE < 30 # by default is an AND (`&&`)
@rsubset df :eGFR > 100 || :AGE < 30 # OR (`||`)

@subset df :SCR .> mean(:SCR) # Why `@rsubset` would fail here?

@subset df begin
    :SCR .> mean(:SCR)
    :eGFR .< median(:eGFR)
end

## @orderby
@orderby df :eGFR  # ascending by default
@orderby df -:eGFR # descending

@orderby df -:ISMALE :eGFR  # several conditions

## @chain
# This is the pipe!
@chain df begin
    @select $(Between(:SCR, :eGFR))
    @transform begin
        :eGFR_z = begin
            μ = mean(:eGFR)
            σ = std(:eGFR)
            [(x - μ) / σ for x in :eGFR]
        end
    end
    @rsubset :eGFR > 100
    @orderby -:ISMALE :eGFR
end

## groupby and @combine
q25(x) = quantile(x, 0.25)
q75(x) = quantile(x, 0.75)
@chain df begin
    groupby(:ISMALE)
    @combine begin
        :AGE_μ = mean(:AGE)
        :WEIGHT_μ = mean(:WEIGHT)
        :WEIGHT_q25 = q25(:WEIGHT)
        :WEIGHT_q75 = q75(:WEIGHT)
        :total = length(:ID)
        :high_eGFR = count(>(80), :eGFR)
    end
end

# Something hard to do in tidyverse
# using a lazy evaluation `$()` with broadcasting `.`
@chain df begin
    groupby(:ISMALE)
    @combine $([:AGE, :WEIGHT] .=> [mean median])
end

# or even a one-liner
@by df :ISMALE $([:AGE, :WEIGHT] .=> [mean median])
