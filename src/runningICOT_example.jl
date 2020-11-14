using DataFrames, MLDataUtils
using Clustering, Distances
using CSV
using Random
using Logging

# Set up Logging - we recommend to use this command to avoid package warnings during the model training process.
logger = Logging.SimpleLogger(stderr,Logging.Warn);
global_logger(logger);

# Read the data - recommend the use of the readtable() command to avoid conflicts with the CSV package.
data = readtable("../data/ruspini.csv"); 

# Convert the dataset to a matrix
data_array = convert(Matrix{Float64}, data);
# The ruspini dataset comes with pre - defined clusters that we will use as a point of reference for the K-means algorithm. 
# Note that otherwise the elbow method is suggested.
K = length(unique(data_array[:,end]))
# Get the number of observations and features
n, p = size(data_array)
data_t = data_array';



#Set parameters for the learners
cr = :dunnindex
method = "ICOT_local"
warm_start = :none;
geom_search = false
threshold = 0.0
seed = 1
gridsearch = false
num_tree_restarts = 100
complexity_c = 0.0
min_bucket = 10
maxdepth = 5


data_array = convert(Matrix{Float64}, data);
K = length(unique(data_array[:,end]))
n, p = size(data_array)
data_t = data_array';

@test n==size(data_t,2)
@test p==size(data_t,1)

#Fix the seed
Random.seed!(seed);

#Get the assignments from kmeans
kmeans_result = kmeans(data_t, K);
assignment = kmeans_result.assignments;
@test K==length(unique(assignment))

data_full = DataFrame(hcat(data, assignment, makeunique=true));
names!(data_full, [:x1, :x2, :true_labels, :kmean_assign]);
# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)

X = data_full[:,1:2]; y = data_full[:,:true_labels];

@test size(X,2)==p-1
@test K==length(unique(y))

## start by testing license
lnr_oct = ICOT.IAI.OptimalTreeClassifier(localsearch = false, max_depth = maxdepth,
													 minbucket = min_bucket,
													 criterion = :misclassification
													 )
grid = ICOT.IAI.GridSearch(lnr_oct)
ICOT.IAI.fit!(grid, X, y)
ICOT.IAI.showinbrowser(grid.lnr)

### ----------------- RUN ICOT ----------------- ###
# include("ICOT.jl")
# Create the local search learner - greedy warm start
warm_start= :greedy
lnr_ws_greedy = ICOT.InterpretableCluster(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, cp = complexity_c, max_depth = maxdepth,
	minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr, kmeans_warmstart = warm_start,
	geom_search = geom_search, geom_threshold = threshold);
run_time_icot_ls_greedy = @elapsed ICOT.fit!(lnr_ws_greedy, X, y);

ICOT.showinbrowser(lnr_ws_greedy)

score_ws_greedy = ICOT.score(lnr_ws_greedy, X, y);
score_al_ws_greedy = ICOT.score(lnr_ws_greedy, X, y, criterion=:silhouette);

@test score_ws_greedy ≈ lnr_ws_greedy.tree_.dunnindex_score atol=1e-8
@test score_al_ws_greedy ≈ lnr_ws_greedy.tree_.silhouette_score atol=1e-8
@test score_ws_greedy ≈ 0.5210140254379996 atol=1e-8
@test score_al_ws_greedy ≈ 0.7399381248816937 atol=1e-8

# Create the local search learner -  oct warm start
warm_start= :oct
lnr_ws_oct = ICOT.InterpretableCluster(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, cp = complexity_c, max_depth = maxdepth,
	minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr, kmeans_warmstart = warm_start,
	geom_search = geom_search, geom_threshold = threshold);
run_time_icot_ls_oct = @elapsed ICOT.fit!(lnr_ws_oct, X, y);

score_ws_oct = ICOT.score(lnr_ws_oct, X, y);
score_al_ws_oct = ICOT.score(lnr_ws_oct, X, y, criterion=:silhouette);

@test score_ws_oct ≈ lnr_ws_oct.tree_.dunnindex_score atol=1e-8
@test score_al_ws_oct ≈ lnr_ws_oct.tree_.silhouette_score atol=1e-8
@test score_ws_oct ≈ 0.5210140254379996 atol=1e-8
@test score_al_ws_oct ≈ 0.7399381248816937 atol=1e-8

