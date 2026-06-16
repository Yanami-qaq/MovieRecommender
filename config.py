"""Central configuration for the recommendation pipeline."""

# Data filtering
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 10
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Model hyperparameters
N_NEIGHBORS = 30
SVD_N_FACTORS = 50
SVD_N_EPOCHS = 20
SVD_LR = 0.005
SVD_REG = 0.02
HYBRID_CF_WEIGHT = 0.6
HYBRID_CB_WEIGHT = 0.4

# Evaluation
EVAL_K = 10
EVAL_N_USERS = 200
RATING_THRESHOLD = 4.0

# Paths
FIGURES_DIR = "data/processed/figures"
LOG_DIR = "data/processed/logs"
