import get_link_prediction
import pandas as pd
import networkx as nx


author_id = '617101adfaad3d5fe1840a25'

# Link df
link_df = pd.read_csv('data/cleaned_df.csv', index_col=0)
# Authors df
authors_df = pd.read_csv('data/final_3_pca.csv', index_col=0)
# Authors_info df
authors_info_df = pd.read_csv('data/Author_Institute_Lat_Long_Topic.csv', index_col=0)

# Load the graph.
G_train, G_test = get_link_prediction.load_graph()

# Create the model.
model = get_link_prediction.create_model(G_train)

# Load the weights of the model.
model = get_link_prediction.load_weights(model)

# Get the link prediction results.
top_10_predictions = get_link_prediction.get_link_prediction_results(model, author_id, G_test)

authors_list = [author_id] + top_10_predictions
# subset the link_df source or target is in authors_list
link_df_subset = link_df[link_df['source'].isin(authors_list) | link_df['target'].isin(authors_list)]

# Add 10 links
link_df_subset = link_df_subset.append(pd.DataFrame({'source': [author_id]*10, 'target': top_10_predictions}))

# Name of the nodes
Name_dict = authors_info_df.set_index('Author')['Name'].to_dict()

# Topic
authors_df_subset = authors_df[authors_df.index.isin(authors_list)]
Topic_dict = authors_df_subset['Topic'].to_dict()


# Create the graph
G = nx.from_pandas_edgelist(link_df_subset, 'source', 'target', create_using=nx.Graph())

# Add node features to the graph
for node in G.nodes():
    try:
        G.nodes[node]['pca'] = authors_df.loc[node].values.tolist()
    except:
        G.nodes[node]['pca'] = [0]*len(authors_df.columns)

# Add node names to the graph
for node in G.nodes():
    try:
        G.nodes[node]['Name'] = Name_dict[node]
    except:
        G.nodes[node]['Name'] = 'Unknown'

# Add node topics to the graph
for node in G.nodes():
    try:
        G.nodes[node]['Topic'] = str(Topic_dict[node])
    except:
        G.nodes[node]['Topic'] = 'Unknown'


from pyvis.network import Network

# set the physics layout of the network
net = Network(bgcolor="#222222", font_color="white")

net.force_atlas_2based()


# add nodes to the network
for node in G.nodes():
    net.add_node(node, label=G.nodes[node]['Name'], title=G.nodes[node]['Topic'])

# add edges to the network
for edge in G.edges():
    net.add_edge(edge[0], edge[1])

# show the visualization
net.show('test.html')

# save the network
net.save_graph('test.html')







