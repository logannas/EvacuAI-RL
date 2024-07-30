import random
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque
from loguru import logger
from typing import Dict, List, Tuple, Union
import pickle

from utils.create_graph import create_graph
from model.agent import Agent


class GraphDeepRL:
    def __init__(
        self,
        transfer_learning: bool = False,
        transfer_learning_path: Union[bool, None] = None,
        hyp: dict = {
            "beta": 1,
            "lr": 1e-2,
            "batchsize": 240,
            "buffer_size": 10**6,
            "episodes": 6000,
            "gamma": 0.99,
            "epsilon": 0.5,
        },
        fire: list = [],
        exit: list = [],
        graph_position: dict = {},
        edges: list = [],
        previous_state: Union[int, None] = None,
    ):
        self.previous_state = previous_state
        logger.info(f"Number of edges: {len(edges)}")

        hyp["episodes"] = int((len(edges) / 2) * 500)

        if len(edges) > 80:
            hyp["episodes"] = int((len(edges) / 2) * 900)

        if previous_state:
            hyp["episodes"] = int((len(edges) / 2) * 50)

        if transfer_learning:
            print(transfer_learning)
            hyp["episodes"] = int((len(edges) / 6) * 500)
            logger.info(f"Cuda is available: {T.cuda.is_available()}")
            transfer_learning_path = pickle.loads(transfer_learning_path)
            logger.info(transfer_learning_path)

        graph = create_graph(edges, graph_position, exit, fire)
        type_graph = type(graph)
        print(f"Graph: {type_graph}")

        logger.info("Initialize Agent")
        self.agent = Agent(graph, fire, exit, transfer_learning_path, hyp)

    def train(self):
        return self.agent.train(self.previous_state)

    def inference(self):
        return self.agent.inference(self.previous_state)


graph_model = GraphDeepRL(
    transfer_learning=False,
    fire=[8, 9, 10],
    exit=[0, 12],
    graph_position={
        0: {"x": 285, "y": 300},
        1: {"x": 285, "y": 150},
        2: {"x": 195, "y": 150},
        3: {"x": 210, "y": 195},
        4: {"x": 210, "y": 165},
        5: {"x": 210, "y": 180},
        6: {"x": 225, "y": 210},
        7: {"x": 225, "y": 225},
        8: {"x": 390, "y": 150},
        9: {"x": 375, "y": 180},
        10: {"x": 360, "y": 195},
        11: {"x": 360, "y": 225},
        12: {"x": 450, "y": 300},
        13: {"x": 390, "y": 210},
        14: {"x": 450, "y": 255},
        15: {"x": 390, "y": 60},
        16: {"x": 345, "y": 60},
        17: {"x": 285, "y": 60},
        18: {"x": 240, "y": 60},
        19: {"x": 195, "y": 60},
        20: {"x": 90, "y": 90},
        21: {"x": 60, "y": 90},
        22: {"x": 75, "y": 180},
        23: {"x": 150, "y": 210},
        24: {"x": 195, "y": 285},
    },
    edges=[
        (0, 1, 150.0),
        (1, 0, 150.0),
        (1, 2, 90.0),
        (2, 1, 90.0),
        (1, 4, 76.48529270389177),
        (4, 1, 76.48529270389177),
        (1, 5, 80.77747210701756),
        (5, 1, 80.77747210701756),
        (1, 3, 87.46427842267951),
        (3, 1, 87.46427842267951),
        (1, 6, 84.8528137423857),
        (6, 1, 84.8528137423857),
        (1, 7, 96.04686356149273),
        (7, 1, 96.04686356149273),
        (7, 0, 96.04686356149273),
        (0, 7, 96.04686356149273),
        (6, 0, 108.16653826391968),
        (0, 6, 108.16653826391968),
        (3, 0, 129.0348790056394),
        (0, 3, 129.0348790056394),
        (5, 0, 141.50971698084905),
        (0, 5, 141.50971698084905),
        (4, 0, 154.434452114805),
        (0, 4, 154.434452114805),
        (2, 0, 174.92855684535903),
        (0, 2, 174.92855684535903),
        (1, 8, 105.0),
        (8, 1, 105.0),
        (1, 9, 94.86832980505137),
        (9, 1, 94.86832980505137),
        (1, 10, 87.46427842267951),
        (10, 1, 87.46427842267951),
        (1, 11, 106.06601717798213),
        (11, 1, 106.06601717798213),
        (13, 8, 60.0),
        (8, 13, 60.0),
        (9, 13, 33.54101966249684),
        (13, 9, 33.54101966249684),
        (10, 13, 33.54101966249684),
        (13, 10, 33.54101966249684),
        (11, 13, 33.54101966249684),
        (13, 11, 33.54101966249684),
        (13, 14, 75.0),
        (14, 13, 75.0),
        (14, 12, 45.0),
        (12, 14, 45.0),
        (8, 0, 183.09833423600554),
        (0, 8, 183.09833423600554),
        (9, 0, 150.0),
        (0, 9, 150.0),
        (10, 0, 129.0348790056394),
        (0, 10, 129.0348790056394),
        (11, 0, 106.06601717798213),
        (0, 11, 106.06601717798213),
        (1, 19, 127.27922061357856),
        (19, 1, 127.27922061357856),
        (18, 1, 100.62305898749054),
        (1, 18, 100.62305898749054),
        (17, 1, 90.0),
        (1, 17, 90.0),
        (16, 1, 108.16653826391968),
        (1, 16, 108.16653826391968),
        (15, 1, 138.2931668593933),
        (1, 15, 138.2931668593933),
        (21, 22, 91.2414379544733),
        (22, 21, 91.2414379544733),
        (20, 22, 91.2414379544733),
        (22, 20, 91.2414379544733),
        (22, 23, 80.77747210701756),
        (23, 22, 80.77747210701756),
        (23, 24, 87.46427842267951),
        (24, 23, 87.46427842267951),
        (24, 0, 91.2414379544733),
        (0, 24, 91.2414379544733),
    ],
)

train = graph_model.train()
inference = graph_model.inference()
print(inference)
