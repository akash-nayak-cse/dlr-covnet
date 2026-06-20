"""
DLR-CovNet Configuration
------------------------
Centralized model and training hyperparameters used in the published
DLR-CovNet architecture.

Note:
This repository accompanies the Springer publication and is intended for
research demonstration and portfolio purposes.
"""

# Model Configuration
INPUT_SHAPE = (224, 224, 1)

# Training Configuration
BATCH_SIZE = 64
EPOCHS = 14
LEARNING_RATE = 1e-5
RANDOM_SEED = 42

# Optimization
EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE = 6
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-7