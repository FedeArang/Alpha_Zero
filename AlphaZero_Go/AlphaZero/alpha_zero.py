from Go.gotypes import Player
from Go.board_fast import GameState
from Go import scoring
from AlphaZero.agent import Agent
from Go import board_encoder
import numpy as np
#from keras.optimizers import SGD
#import torch
import torch.nn.functional as F

from tqdm import trange
import experience


class AlphaZero():

    def __init__(self, model, hyperparams):
        self.model = model
        self.hyperparams = hyperparams

    def self_play(self, black_player, white_player, board_size):

        moves_played=[]
        game_state=GameState.new_game(board_size)
        agents={Player.black: black_player, Player.white: white_player}

        while not game_state.is_over():

            next_move=agents[game_state.next_player].select_move(game_state)
            moves_played.append(next_move)
            game_state=game_state.apply_move(next_move)

        game_result = scoring.compute_game_result(game_state)

        return moves_played, game_result.winner, game_result.winning_margin


    def train(self, model, experience_buffer, optimizer):

        n_samples = experience_buffer.states.shape[0]

        model_input = experience_buffer.states

        # OBSERVATION: the target policy, which we will use to train the network, are the visit frequencies/counts
        # that we calculated in the tree search. Intuitevely, this is because as we become better in the search,
        # then the visit counts correspond to the optimal policy
        total_visits = np.sum( experience_buffer.visit_counts, axis=1).reshape((n_samples, 1))
        action_target = experience_buffer.visit_counts / total_visits

        value_target = experience_buffer.rewards

        out_policy, out_value=model(model_input) #calculate the predictions of the model

        optimizer=optimizer

        #calculate the losses wrt value and policy and sum them together to get the overall loss
        policy_loss=F.cross_entropy(out_policy, action_target)
        value_loss=F.mse_loss(out_value, value_target)
        loss=policy_loss+value_loss

        #optimization step/weights update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''self.model.compile(
            SGD(lr= self.hyperparams['learning_rate']),
            loss=['categorical_crossentropy', 'mse'])
        self.model.fit(
            model_input, [action_target, value_target],
            batch_size= self.hyperparams['batch_size'])'''


    def learning_schedule(self, board_size, optimizer):


        for iteration in range(self.hyperparams['num_iterations']):

            agent1 = Agent(self.model, board_encoder, self.hyperparams)
            agent2 = Agent(self.model, board_encoder, self.hyperparams)

            # initialize the experience collector
            c1 = experience.ZeroExperienceCollector()
            c2 = experience.ZeroExperienceCollector()
            agent1.set_collector(c1)
            agent2.set_collector(c2)

            # we turn the model to evaluation since in the game function we need to
            # evaluate/make inferences without doing backprop

            color1 = Player.black

            self.model.eval()

            for game_iteration in trange(self.hyperparams['num_game_iterations']):

                c1.begin_episode()
                agent1.set_collector(c1)
                c2.begin_episode()
                agent2.set_collector(c2)

                if color1 == Player.black:
                    black_player, white_player = agent1, agent2
                else:
                    white_player, black_player = agent2, agent1

                # in each game iteration, we collect new data (in particular, we play a single game)
                moves, winner, winning_margin = self.self_play(black_player, white_player, board_size)

                #we give to the winner a reward of +1 and to the loser a reward of -1
                if winner == color1:
                    c1.complete_episode(reward=1)
                    c2.complete_episode(reward=-1)
                else:
                    c2.complete_episode(reward=1)
                    c1.complete_episode(reward=-1)

                #at next episode, agent2 will be black and viceversa (we alternate at each game)
                color1 = color1.other

            #we combine the two experience collections
            #indeed, notice that both agents use the same model, hence we are going to train a single model and
            #we are going to use both "perspectives" of the game
            experience_collection = experience.combine_experience([c1, c2])

            # we turn back the model to training mode since now we are ready to perform backprop

            #self.model.train(experience_collection) we do not need this if we use Keras

            self.model.train()

            for epoch in trange(self.hyperparams['num_epochs']):
                self.train(self.model, experience_collection, optimizer)


