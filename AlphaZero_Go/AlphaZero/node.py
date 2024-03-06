

class Branch():

    def __init__(self, prior):
        self.prior=prior
        self.visit_count=0
        self.total_value=0.0

class Node():

    def __init__(self, game_state, parent_node, previous_move, value, priors):

        #game state: the current board position
        #parent_node: we keep track of the parent node, since we will need this when we will do backpropagation of values and visit counts
        #previous_move: we also keep track of the previous move
        #value: value of the node
        #priors: a dictionary that assigns to each move (key) a probability

        self.game_state=game_state
        self.parent_node=parent_node
        self.previous_move=previous_move
        self.value=value
        self.total_visit_count=1 #when we initialize the node, it means that we have visited it for the first time, hence we set the visit count to 1

        #   TODO Is it really necessary to have children and branches separately? Could we simply store the branches attributes in the node itself?
        self.branches={}

        #TODO we are not renormalizing the probabilities but we just limit to discard some of them? WOuld it be better if we renormalize them?
        for move, p in priors.items():
            if game_state.is_valid_move(move):
                self.branches[move]=Branch(p) #if the move is valid, then we instantiate a new branch

        self.children={}

    def get_expected_value(self, move):

        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count #we return the weighted value of the branch

    def get_prior(self, move):
        return self.branches[move].prior

    def get_visit_count(self, move):

        if move in self.branches:
            return self.branches[move].visit_count
        return 0

    def get_moves(self):
        return self.branches.keys()

    def update_branch(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]



