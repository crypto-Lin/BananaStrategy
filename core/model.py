# xgb params intro:
# Parameters for tree booster
# eta/learning rate
# gamma/min_split_loss : Minimum loss reduction required to make a further partition on a leaf node of the tree.
# The larger gamma is, the more conservative the algorithm will be.
# min_child_weight : Minimum sum of instance weight (hessian) needed in a child.
# If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
# then the building process will give up further partitioning.
# The larger min_child_weight is, the more conservative the algorithm will be. range: [0,∞]
# max_delta_step : it might help in logistic regression when class is extremely imbalanced.
# Set it to value of 1-10 might help control the update.
# subsample : Subsample ratio of the training instances. this will prevent overfitting.
# Subsampling will occur once in every boosting iteration.
# tree_method : string [default= auto]
# params = auto, exact, approx, hist, gpu_hist, the larger the data, the righter method to choose
# scale_pos_weight : Control the balance of positive and negative weights, useful for unbalanced classes.
# num_parallel_tree, [default=1] - Number of parallel trees constructed during each iteration.
# This option is used to support boosted random forest.

# Learning task parameters
#