import networkx as nx


G = nx.Graph(day='Sunday') # attribute to graph
G.graph['day'] = 'Wednesday' # modify graph attribute 
print(G.graph)


# Node attribute 
G.add_node(1, time = '5am')
G.add_node(2, time = '6pm')
print(G.nodes[2])

# Add node with same attributes 
myList = list(range(3, 6))
G.add_nodes_from(myList, time = '3pm')


# Modify node attribute 
G.nodes[1]['time'] = '4pm'
print(G.nodes[1])


# Multiple attributes in a node/graph 
G.nodes[1]['day'] = '4th May'
print(G.nodes[1])


# Edge attribute 
G.add_edge(1, 2, weight = 4.5) 
print(G[1][2])

import dgl.data

dataset = dgl.data.MUTAGDataset()
graph = dataset[0]


data = dgl.data.GINDataset(name='PTC', self_loop=False)
g, label = data[128]








