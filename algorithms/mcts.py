from __future__ import division

import math
import random
from pprint import pprint

def randomPolicy(original_state, discount):
    state = original_state.get_copy()
    
    reward = state.score
    steps = 0
    
    while not state.terminal:
        try:
            action = random.choice(state.actions)
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        last_game_score = state.score
        state.step(action)
        step_reward = state.score - last_game_score - 0.1
        reward += step_reward * (discount ** steps)
        steps += 1
    
    return reward


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.terminal
        self.parent = parent
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        
        self.numVisits = 0
        self.value = 0
        self.children = {}
        self.ucb = float("-inf")

    def __str__(self):
        s = f"R: {round(self.value, 2)}, V: {self.numVisits}, U: {round(self.ucb, 2)}"
        return s

class MCTS():
    def __init__(self, iterations=1000, explorationConstant=2, discount = 0.99, rolloutPolicy=randomPolicy, verbose=False):

        self.verbose = verbose
        
        self.iterations = iterations
        self.explorationConstant = explorationConstant
        self.discount = discount
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)
        
        for i in range(self.iterations):
            self.execute_iteration()

        action = max([(action, node.value) for action, node in self.root.children.items()],key=lambda x:x[1])[0]
        return action
    
    def get_info(self):
        info = ""
        for action, node in self.root.children.items():
            info += f"{action}: {str(node)} /// "
        return info

    def execute_iteration(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        if self.verbose:
            print("######## Root State ##########")
            pprint(self.root.state.grid)
            print(self.get_info())
            node = self.selectNode(self.root)
            print(self.get_info())
            node = self.expand(node)
            print(self.get_info())
            reward = self.rollout(node.state, self.discount)
            print(self.get_info())
            self.backpropogate(node, reward)
            print(self.get_info())
        else:
            node = self.selectNode(self.root)
            node = self.expand(node)
            reward = 0 if node.is_terminal else self.rollout(node.state, self.discount)
            self.backpropogate(node, reward)

    def selectNode(self, node):
        search_node = node
        is_expandable = len(search_node.children) < len(search_node.state.actions)
        while not (search_node.is_terminal or is_expandable):
            search_node = self.getBestChild(search_node, self.explorationConstant)
            is_expandable = len(search_node.children) < len(search_node.state.actions)
        return search_node

    def expand(self, node):
        if node.is_terminal:
            return node
        actions = node.state.actions
        for action in actions:
            if action not in node.children:
                state_copy = node.state.get_copy()
                state_copy.step(action)
                newNode = treeNode(state_copy, node)
                node.children[action] = newNode
                # print(f"exploring action: {action}")
                return newNode

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.value += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        best_ucb = float("-inf")
        best_nodes = []
        for action, child in node.children.items():
            child.ucb = child.value / child.numVisits + explorationValue * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            if child.ucb > best_ucb:
                best_ucb = child.ucb
                best_nodes = [child]
            elif child.ucb == best_ucb:
                best_nodes.append(child)
        return random.choice(best_nodes)