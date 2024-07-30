import random
import networkx as nx
import torch as T

from typing import Dict, List, Tuple


class Environment(object):
    def __init__(
        self,
        G: nx.classes.graph.Graph,
        exit: List,
        fire: List,
        device: T.device,
        beta: float,
    ) -> None:
        self.device = device
        self.graph = G
        self.exit = exit
        self.fire = fire
        self.beta = beta

        node_list = list(self.graph.nodes)
        self.adj_mat = nx.adjacency_matrix(self.graph, nodelist=node_list).todense()
        # print(f'Adjacency Matrix: {self.adj_mat}')

        self.num_nodes = len(self.graph.nodes)
        self.n_actions = self.num_nodes
        self.obs_size = self.num_nodes * 2
        self.dict_node = self.create_dict()

    def create_dict(self) -> Dict:
        dict_node = {}

        for idx, node in enumerate(self.graph.nodes):
            dict_node[node] = idx

        return dict_node

    def reward(self, current_node: int, new_node: int) -> Tuple[float, bool]:
        connected = False

        if new_node in self.fire:
            rw = -1000

        elif new_node in self.graph[current_node]:
            # rw = self.graph[current_node][new_node]['weight']* 0.1
            rw = self.adj_mat.item((current_node, new_node)) * 0.1
            connected = True

        else:
            rw = -100

        return rw, connected

    def call_reward(self, current_node: int, action: int) -> Tuple[int, bool]:
        discount = self.beta

        current_node = self.dict_node[current_node]
        new_node = self.dict_node[action]
        rw, connected = self.reward(current_node, new_node)

        function_reward = rw * discount

        return function_reward, connected

    def next_state(self, action: int) -> int:
        nodes = self.num_nodes

        return action + nodes

    def current_node_end(self, state: int) -> Tuple[int, int]:
        nodes = self.num_nodes
        destination = state % nodes
        end = (state - destination) / nodes

        return destination, int(end)

    def step(self, state: int, action: int) -> Tuple[int, int, bool]:
        done = False

        current_node, _ = self.current_node_end(state)
        new_state = self.next_state(action)

        reward, connected = self.call_reward(current_node, action)

        if not connected:
            new_state = state

        elif action in self.exit:
            reward = 10000
            done = True

        return new_state, reward, done

    def define_start(self) -> int:
        while True:
            start = random.randint(0, self.num_nodes - 1)
            if start not in self.exit:
                break

        return start

    def reset(self, previous_state=None) -> int:
        if previous_state:
            start = int(previous_state)

        else:
            start = self.define_start()

        state = self.next_state(self.dict_node[start])

        return state

    def state_to_vector(self, current_node: int, end_node: int) -> List:
        n_nodes = len(self.graph.nodes)

        source_list_zeros = [0.0] * n_nodes
        source_list_zeros[current_node] = 1

        end_list_zeros = [0.0] * n_nodes
        end_list_zeros[end_node] = 1.0

        vector = source_list_zeros + end_list_zeros

        return vector

    def list_of_vectors(self, new_states_t: T.Tensor) -> List:
        list_new_states_t = new_states_t.tolist()
        list_new_states_t = [int(v) for v in list_new_states_t]

        vector_list = []
        for state in list_new_states_t:
            s, f = self.current_node_end(state)
            vector = self.state_to_vector(s, f)
            vector_list.append(vector)

        return vector_list
