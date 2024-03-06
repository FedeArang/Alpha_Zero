import torch
import numpy as np
from AlphaZero.node import Node, Branch

class Agent():

    def __init__(self, model, encoder, hyperparams):

        self.model=model
        self.encoder=encoder #encoder of position
        self.hyperparams=hyperparams #TODO include list of hyperparameters in the comments
        self.collector = None


    @torch.no_grad
    def tree_search(self, starting_state):

        #starting_state: the board position from which we will begin the tree search from

        root=self.generate_node(starting_state) #based on the starting state, we create the starting node (i.e. we include the priors, values, branches, etc.)

        for _ in range(self.hyperparams['n_simulations']):

            node = root #at the beginning of the simulation, we set the node to be the root
            next_move = self.select_branch(node) #we select next move according to the UCT formula

            while node.has_child(next_move): #we continue the exploration untile the current node does not have any children

                node=node.get_child(next_move) #go to the node corresponding to the selected move
                next_move=self.select_branch(node) #select next move from new node

            #once we finished the simulation and we arrived at a leaf with no childre, we build a new
            #node corresponding to the children of the leaf corresponding to the last move

            new_state=node.game_state.apply_move(next_move)
            new_node=self.generate_node(new_state, parent_node=node, previous_move=next_move)

            #now we are ready to update the statistics, in particular visit counts and the total values by backpropagating
            #through all the visited nodes in the simulaiton

            self.update_statistics(next_move, new_node)

            '''if self.collector is not None:
                root_state_tensor = self.encoder.encode(starting_state)
                visit_counts = np.array([
                    root.visit_count(
                        self.encoder.decode_move_index(idx))
                    for idx in range(self.encoder.num_moves())
                ])
                self.collector.record_decision(
                    root_state_tensor, visit_counts)'''



    @torch.no_grad
    def select_move(self, starting_state):

        root=self.tree_search(starting_state)
        return max(root.get_moves(), key=root.visit_count)


    @torch.no_grad
    def generate_node(self, game_state,  parent_node=None, previous_move=None):

        num_moves = game_state.board.num_rows * game_state.board.num_cols

        encoded_state=self.encoder.encode(game_state) #we encode the board position in the format we want to give as input to the model
        model_input = np.array([encoded_state])
        policy, values=self.model.predict(model_input) #we predict, using the current model, the policy (i.e. the probability of choosing moves) and the values
        policy = policy[0]

        if np.random.random() < self.hyperparams['temperature']:
            # Explore random moves.
            policy = np.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            policy = policy

        #clip policy probabilities
        eps = 1e-5
        policy = np.clip(policy, eps, 1 - eps)

        value = values[0][0]

        #if we are at the root, we add dirichlet noise to encourage exploration
        if parent_node is None:
            policy=(1 - self.hyperparams['dirichlet_epsilon']) * policy + self.hyperparams['dirichlet_epsilon'] \
            * np.random.dirichlet([self.hyperparams['dirichlet_alpha']] * num_moves)

        #TODO again, here we do not check for valid moves and we do not reweight the policy accordingly.. will have to see if it still works

        policy_dict=dict() #we create a dictionary to store the policy, where the keys are the moves corresponding to the probabilities

        for i, prob in enumerate(policy):
            policy_dict[self.encoder.decode_move_index(i)]=prob

        new_node=Node(game_state, parent_node, previous_move, value, policy) #we create an instance of the new node we are going to create

        if parent_node is not None:
            parent_node.children[previous_move]=new_node #if we are not at the root, we add the new node we have just created to the parent node's children

        return new_node


    def select_branch(self, node):

        #given a node, we select the next move based on the UCT formula when we deploy simulations

        total_count = node.total_visit_count

        def score_branch(move):

            q=node.get_expected_value(move)
            n=node.get_visit_count(move)
            p=node.get_prior(move)

            return q + self.hyperparams['c'] * p * np.sqrt(total_count) / (n + 1)

        return max(node.get_moves(), key=score_branch)


    def update_statistics(self, move, node):

        value=-1*node.value #we need to take the opponents perspective

        #we keep updating the statistics until we get to the root
        while node is not None:

            node.update_branch(move, value)
            move = node.last_move
            node = node.parent
            value = -1 * value


    def set_collector(self, collector):
        self.collector = collector








