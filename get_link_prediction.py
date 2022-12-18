"""
This script is used to get the link prediction results of the trained model.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
import networkx as nx


# ==============================================================================


def load_graph():
    """
    Load the graph.

    Returns:
        G_train : (stellargraph.StellarGraph) The graph.
        G_test : (stellargraph.StellarGraph) The graph.
    """
    path_link_df = "data/cleaned_df.csv"
    link_df = pd.read_csv(path_link_df, index_col=0)

    path_author_df = "data/final_3_pca.csv"
    authors = pd.read_csv(path_author_df, index_col=0)

    G_nx = nx.from_pandas_edgelist(link_df, source="source", target="target", create_using=nx.Graph())

    # Add node features to the graph
    for node in G_nx.nodes():
        try:
            G_nx.nodes[node]['pca'] = authors.loc[node].values.tolist()
        except:
            G_nx.nodes[node]['pca'] = [0]*len(authors.columns)
    
    # Convert the graph to StellarGraph
    G = StellarGraph.from_networkx(G_nx, node_features="pca")

    # Define an edge splitter on the original graph G:
    edge_splitter_test = EdgeSplitter(G)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, _, _ = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True)

    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
    # reduced graph G_train with the sampled links removed:
    G_train, _, _ = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    return G_train, G_test, G_nx


# ==============================================================================


def create_model(G_train):
    """
    Create the model.

    Args:
        G_train : (stellargraph.StellarGraph) The graph.

    Returns:
        model : (keras.Model) The model.
    """
    batch_size = 20
    num_samples = [20, 10]
    train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
    # Define the model.
    graphsage = GraphSAGE(
        layer_sizes=[20, 20], generator=train_gen, bias=True, dropout=0.3)
    x_inp, x_out = graphsage.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    model = tf.keras.Model(inputs=x_inp, outputs=prediction)

    # Compile the model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["acc"],
    )

    return model


# ==============================================================================


def load_weights(model):
    """
    Load the weights of the model.

    Args:
        model : (keras.Model) The model.
    """
    # Load the weights of the model.
    model_path = "models/weights.h5"
    model.load_weights(model_path)

    return model


# ==============================================================================


def get_link_prediction_results(model, author_id, G_test, G_nx):
    """
    Get the link prediction results.

    Args:
        model : (keras.Model) The model.
        author_id : (int) The author id.

    Returns:
        link_prediction_results : (numpy.ndarray) The link prediction results.
    """
    #Get edge between author_id and not connected nodes
    edges = []
    for node in G_test.nodes():
        if node != author_id and not G_nx.has_edge(author_id, node):
            edges.append((author_id, node))

    # Convert to ndarray
    edges_array = np.array(edges)
    # Dummy labels
    labels = np.zeros(len(edges))

    batch_size = 20
    num_samples = [20, 10]


    test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
    test_flow = test_gen.flow(edges_array, labels)

    predicted = model.predict(test_flow, workers=1, verbose=0)
    predicted = predicted.squeeze()
    predicted = predicted.tolist()


    # Get map predict to edges
    map_predict_to_edges = {}
    for i in range(len(predicted)):
        map_predict_to_edges[predicted[i]] = edges[i]
    # Sort the map by key
    map_predict_to_edges = dict(sorted(map_predict_to_edges.items(), key=lambda item: item[0], reverse=True))

    # Get the top 10 predictions
    top_10_predictions = []
    for i, key in enumerate(map_predict_to_edges):
        temp = list(map_predict_to_edges[key])[1]
        if i < 10:
            top_10_predictions.append(temp)
    
    return top_10_predictions


# =================================================================================