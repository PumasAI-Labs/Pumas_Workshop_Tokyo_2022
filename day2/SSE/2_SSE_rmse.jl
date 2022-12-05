
##################################################################
### Instructions:
###     1) Change to this directory
###     2) Upload the results from the previous run to the folder
###     3) Run the code below in the interactive session
##################################################################

using Serialization

##################################################################
### Find the jls after uploading it
##################################################################
result_filename = first(filter(f->contains(f, ".jls"), readdir()))

##################################################################
### Deserialize
##################################################################
results = deserialize(result_filename)

##################################################################
### Grab all the vectors of fits
##################################################################
fits = map(r->r.fits, results)

##################################################################
### Calculate RMSE for θ_CL_parent and scatterplot against dose 
##################################################################
rmse = map(r -> sqrt(mean(map(r_j -> (r_j.θ_CL_parent - first(r).param.θ_CL_parent)^2, coef.(r)))), fits)
