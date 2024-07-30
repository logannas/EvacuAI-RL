import networkx as nx
from loguru import logger
import matplotlib.pyplot as plt


def create_graph(edges, position, exit, fire):
    g = nx.Graph()

    for node in position:
        current_position = (position[node]["x"], position[node]["y"])
        g.add_node(node, pos=current_position)

    g.add_weighted_edges_from(edges)
    pos = nx.get_node_attributes(g, "pos")

    color_map = []
    for node in g:
        if node in exit:
            color_map.append("green")
        elif node in fire:
            color_map.append("red")
        else:
            color_map.append("blue")

    nx.draw(g, pos, node_color=color_map, with_labels=True)

    plt.title("Graph")
    plt.show()
    plt.clf()

    logger.info("Total numbers of nodes: " + str(len(g.nodes)))
    return g
