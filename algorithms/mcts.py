from __future__ import annotations
import math
import random

class SnakeEnv:
    pass

class Node():
    def __init__(self, state: SnakeEnv, parent: Node):
        self.state = state
        self.is_terminal = state.terminal
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        
        self.num_visits = 0
        self.value = 0
        self.children = {}
        self.ucb = float("-inf")

class MCTS():
    def __init__(self, iterations=1000, exploration_constant=2, discount = 0.99, step_cost=0.5):        
        self.iterations = iterations
        self.C = exploration_constant
        
        # Bonus: it is good practice to add a discount to diminish value of actions further out in the future
        self.discount = discount
        # Bonus: adding step cost will ensure that the algorithm does not prefer driving around surviving over risking to eat
        self.step_cost = step_cost

    def find_best_action(self, current_state: SnakeEnv):
        # Initialize root with current state
        root = Node(current_state, None)
        
        for _ in range(self.iterations):
            # Execute 4 steps of one iteration to update the tree
            selected_node = self.__selection(root)
            added_node = self.__expand(selected_node)
            reward = self.__simulation(added_node)
            self.__backpropogation(added_node, reward)

        # After iteration budget is used, select best action from root node
        action = max([(action, node.value) for action, node in root.children.items()],key=lambda x:x[1])[0]
        return action
        

    def __selection(self, node: Node) -> Node:
        search_node = node
        
        is_expandable = len(search_node.children) < len(search_node.state.actions)
        # if the selected node can be expanded we stop the search
        while not (search_node.is_terminal or is_expandable):
            best_ucb = float("-inf")
            # track multiple nodes in case of equal UCB values
            best_nodes = []
            for _, child in search_node.children.items():
                # calculate UCB for each child node
                child.ucb = child.value / child.num_visits + self.C * math.sqrt(2 * math.log(search_node.num_visits) / child.num_visits)
                
                if child.ucb > best_ucb:
                    best_ucb = child.ucb
                    best_nodes = [child]
                elif child.ucb == best_ucb:
                    best_nodes.append(child)
            
            # in case of multiple best nodes, randomly choose one
            search_node = random.choice(best_nodes)
            # check if node is a leaf node
            is_expandable = len(search_node.children) < len(search_node.state.actions)
        return search_node

    def __expand(self, node: Node) ->  Node:
        # terminal nodes can not be expanded
        if node.is_terminal:
            return node

        for action in node.state.actions:
            if action not in node.children:
                state_copy = node.state.get_copy()
                state_copy.step(action)
                new_node = Node(state_copy, node)
                node.children[action] = new_node
                return new_node
            
    def __simulation(self, node: Node) -> float:
        if node.is_terminal:
            return 0
        else:
            state = node.state.get_copy()
            reward = state.score
            steps = 0

            while not state.terminal:
                action = random.choice(state.actions)
                last_game_score = state.score
                state.step(action)
                step_reward = state.score - last_game_score - self.step_cost
                reward += step_reward * (self.discount ** steps)
                steps += 1

            return reward

    def __backpropogation(self, node: Node, reward: float) -> None:
        while node is not None:
            node.num_visits += 1
            node.value += reward
            node = node.parent