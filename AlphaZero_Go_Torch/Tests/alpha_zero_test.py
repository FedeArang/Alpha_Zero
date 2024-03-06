#from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input
#from keras.models import Model
from AlphaZero import experience
from AlphaZero.agent import Agent
from AlphaZero.model_torch import ResNet
from Go.board_fast import GameState, Player, Point
from Go import board_encoder
from tqdm import trange
from AlphaZero.alpha_zero import AlphaZero
import torch

def main():
    board_size = 9

    hyperparameters = {
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

    encoder = board_encoder.ZeroEncoder(board_size)

    model =
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    agent1 = Agent(model, board_encoder, hyperparameters)
    agent2 = Agent(model, board_encoder, hyperparameters)

    # initialize the experience collector
    c1 = experience.ZeroExperienceCollector()
    c2 = experience.ZeroExperienceCollector()
    agent1.set_collector(c1)
    agent2.set_collector(c2)

    # we turn the model to evaluation since in the game function we need to
    # evaluate/make inferences without doing backprop

    color1 = Player.black

    model.eval()

    for game_iteration in trange(hyperparameters['num_game_iterations']):

        c1.begin_episode()
        agent1.set_collector(c1)
        c2.begin_episode()
        agent2.set_collector(c2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent2, agent1

        # in each game iteration, we collect new data (in particular, we play a single game)
        moves, winner, winning_margin = AlphaZero.self_play(black_player, white_player, board_size)

        # we give to the winner a reward of +1 and to the loser a reward of -1
        if winner == color1:
            c1.complete_episode(reward=1)
            c2.complete_episode(reward=-1)
        else:
            c2.complete_episode(reward=1)
            c1.complete_episode(reward=-1)

        # at next episode, agent2 will be black and viceversa (we alternate at each game)
        color1 = color1.other

    # we combine the two experience collections
    # indeed, notice that both agents use the same model, hence we are going to train a single model and
    # we are going to use both "perspectives" of the game
    experience_collection = experience.combine_experience([c1, c2])

    # we turn back the model to training mode since now we are ready to perform backprop

    # self.model.train(experience_collection) we do not need this if we use Keras

    model.train()

    for epoch in trange(hyperparameters['num_epochs']):
        AlphaZero.train(model, experience_collection, optimizer)

if __name__ == '__main__':
    main()