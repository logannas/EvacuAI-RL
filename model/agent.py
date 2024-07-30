import random
import time
import networkx as nx
import numpy as np
import torch as T
import torch.nn as nn

from tqdm import tqdm
from collections import deque
from loguru import logger
from typing import Dict, List, Union
import matplotlib.pyplot as plt

from model.environment import Environment
from model.network import Network


class Agent(object):
    def __init__(
        self,
        graph: nx.classes.graph.Graph,
        fire: List = [],
        exit: List = [],
        transfer_learning: Union[str, None] = None,
        hyp: Dict = {},
    ) -> None:
        logger.info(f"Hyperparameters: {hyp}")

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        exit = exit
        fire = fire

        # hyp
        beta = hyp["beta"]
        lr = hyp["lr"]
        self.batch_size = hyp["batchsize"]
        self.episodes = hyp["episodes"]
        self.decay = self.episodes
        self.gamma = hyp["gamma"]
        self.epsilon = hyp["epsilon"]

        self.env = Environment(graph, exit, fire, self.device, beta)
        self.net = Network(self.env.obs_size, self.env.n_actions)
        logger.info(f"Network: {self.net}")

        if transfer_learning:
            logger.info(f"Weights for transfer of learning")
            self.net.load_state_dict(transfer_learning["model"])
            self.net.eval()

        else:
            logger.info("Without Transfer Learning")

        self.target = Network(self.env.obs_size, self.env.n_actions)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = T.optim.Adam(self.net.parameters(), lr=lr)

        self.reward_buffer = [0]
        self.episode_reward = 0.0
        self.buffer_size = hyp["buffer_size"]
        self.min_replay_size = int(self.buffer_size * 0.25)

        self.target_update_frequency = 1000
        self.action_list = np.arange(0, len(self.env.graph.nodes)).tolist()
        self.replay_buffer = deque(maxlen=self.min_replay_size)

    def train(self, previous_state=None) -> None:
        state = self.env.reset(previous_state)

        loss_list = []
        mean_reward = []
        number_ep = []

        state_dict = {"episodes": [], "explore_exploit": [], "time": []}

        path = [state % len(self.env.graph.nodes)]
        for i in tqdm(range(self.episodes)):
            epsilon = np.exp(-i / (self.episodes / 2))
            p = random.random()

            state_dict["episodes"].append(i)

            if p <= epsilon:
                action = np.random.choice(self.action_list)
                state_dict["explore_exploit"].append("explore")

            else:
                current_node, end = self.env.current_node_end(state)
                vector_state = self.env.state_to_vector(current_node, end)
                tensor_state = T.tensor([vector_state])
                action = self.net.act(tensor_state)
                state_dict["explore_exploit"].append("exploit")

            new_state, reward, done = self.env.step(state, action)

            path.append(new_state % len(self.env.graph.nodes))

            # Experience Replay
            transition = (state, action, reward, done, new_state)
            self.replay_buffer.append(transition)
            state = new_state
            self.episode_reward += reward

            if done:
                state = self.env.reset(previous_state)
                self.reward_buffer.append(self.episode_reward)
                self.episode_reward = 0.0
                logger.info(f"Train {i} - Path: {path}")
                path = [state % len(self.env.graph.nodes)]

            if len(self.replay_buffer) < self.batch_size:
                transitions = self.replay_buffer
            else:
                transitions = random.sample(self.replay_buffer, self.batch_size)

            states = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rewards = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            new_states = np.asarray([t[4] for t in transitions])

            states_tensor = T.as_tensor(states, dtype=T.float32).to(self.device)
            actions_tensor = (
                T.as_tensor(actions, dtype=T.int64).to(self.device).unsqueeze(-1)
            )
            rewards_tensor = T.as_tensor(rewards, dtype=T.float32).to(self.device)
            dones_tensor = T.as_tensor(dones, dtype=T.float32).to(self.device)
            new_states_tensor = T.as_tensor(new_states, dtype=T.float32).to(self.device)

            # Target
            list_new_states_tensor = T.tensor(
                self.env.list_of_vectors(new_states_tensor)
            ).to(self.device)
            target_q_values = self.target(list_new_states_tensor)
            max_target_q_values = target_q_values.max(dim=1, keepdim=False)[0]
            targets = (
                rewards_tensor + self.gamma * (1 - dones_tensor) * max_target_q_values
            )
            targets = targets.unsqueeze(-1)

            list_states_tensor = T.tensor(self.env.list_of_vectors(states_tensor)).to(
                self.device
            )
            q_values = self.net(list_states_tensor)
            action_q_values = T.gather(input=q_values, dim=1, index=actions_tensor)

            # Loss MSE
            loss = nn.functional.mse_loss(action_q_values, targets)
            loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_reward.append(np.mean(self.reward_buffer))
            number_ep.append(i)
            dec = {"number_of_episodes": number_ep, "mean_reward": mean_reward}

            if i % self.target_update_frequency == 0:
                self.target.load_state_dict(self.net.state_dict())
            if i % 1000 == 0:
                print("step", i, "avg rew", round(np.mean(self.reward_buffer), 2))
                pass
            if i == 5000:
                pass

        xpoints = np.array(number_ep)
        ypoints = np.array(mean_reward)

        plt.ylabel("Recompensa")
        plt.xlabel("Número de Episódios")
        plt.title("Gráfico Recompensa Acumulada")
        plt.plot(xpoints, ypoints)
        plt.show()
        plt.clf()

        plt.ylabel("Perda")
        plt.xlabel("Número de Episódios")
        plt.title("Gráfico Perda")
        ypoints = np.array(loss_list)

        plt.plot(xpoints, ypoints)
        plt.show()
        plt.clf()

        state = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        return state

    def inference(self, previous_state=None) -> Dict:
        start = time.time()
        model = self.net
        if previous_state:
            dict_nodes = {previous_state: previous_state}
        else:
            dict_nodes = self.env.dict_node.copy()
            for node in self.env.exit:
                dict_nodes.pop(node)

        for node in dict_nodes:
            state = self.env.reset(int(node))
            path = [state % len(self.env.graph.nodes)]

            for i in tqdm(range(self.env.num_nodes)):
                current_node, end = self.env.current_node_end(state)
                vector_state = self.env.state_to_vector(current_node, end)
                tensor_state = T.tensor([vector_state])
                action = model.act(tensor_state)

                new_state, reward, done = self.env.step(state, action)

                path.append(new_state % len(self.env.graph.nodes))

                state = new_state
                self.episode_reward += reward

                if done:
                    dict_nodes[node] = path
                    logger.info(f"Inference {i} - Node: {node} - Path: {path}")
                    break

        result = []
        for i in dict_nodes:
            try:
                init_node = dict_nodes[i][0]
                last_node = dict_nodes[i][-1]
                path = dict_nodes[i]
            except:
                init_node = dict_nodes[i]
                last_node = dict_nodes[i]
                path = dict_nodes[i]

            dict = {
                "init_node": int(init_node),
                "last_node": int(last_node),
                "path": path,
            }
            result.append(dict)

        end = round(time.time() - start, 2)

        return result
