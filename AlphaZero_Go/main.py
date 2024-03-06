from AlphaZero.alpha_zero import AlphaZero
from AlphaZero.model_tf import Model

BOARD_SIZE= 5
hyperparameters= {
    'C': 2,
    'learning_rate': 0.4,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

AZ=AlphaZero(Model, hyperparameters)
AZ.learning_schedule(BOARD_SIZE)


