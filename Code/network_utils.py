import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from fast_tmfg import TMFG

def read_clean_data(path, countries, truncate_end=12):
    """
    Reads and cleans the data from a CSV file.

    Args:
        path (str): The path to the CSV file.
        countries (list): A list of countries to select from the data.
        truncate_end (int, optional): The number of columns to truncate from the end. Defaults to 12.

    Returns:
        tuple: A tuple containing the cleaned data and the dates.
    """
    data = pd.read_csv(path)

    to_drop = ['World Total', 'Other Europe, nes.', 'Areas, not elsewhere specified', 'Rest of America, nes.']
    data = data[~data["ImporterCountry"].isin(to_drop) & ~data["ExporterCountry"].isin(to_drop)] # Remove vague regions
    data = data[(data["ExporterCountry"] != data["ImporterCountry"])] # Remove self-trades

    data = data[(data["ExporterCountry"].isin(countries)) & (data["ImporterCountry"].isin(countries))] # Select countries

    data = data.groupby(["ExporterCountry", "ImporterCountry"]).sum().reset_index() # Aggregate commodities
    data.drop(columns=["Concept", "Commodity", "StartDate", "EndDate"], inplace=True) # Drop unnecessary columns
    data = data.iloc[:, :-truncate_end] # Truncate to 2005-2022

    dates = data.columns[2:].to_list()

    return data, dates

def simple_export_embargo(data, target, init_date, ratio) -> pd.DataFrame:
    """
    Applies a simple export embargo to the given data.

    Args:
        data (DataFrame): The input data containing export information.
        target (str): The target country for the export embargo.
        init_date (str): The initial date from which the embargo starts.
        ratio (float): The reduction ratio applied to the target country's exports.

    Returns:
        DataFrame: The modified data after applying the export embargo.
    """
    data = data.copy()
    dates = data.columns[2:]
    date_idx = dates.get_loc(init_date)

    # Split the data
    data_target = data[data["ExporterCountry"] == target]
    data_others = data[(data["ExporterCountry"] != target) & (data["ImporterCountry"] != target)]

    # Compute total reductions caused by the shock
    reductions = data_target.iloc[:, 2:].sum(axis=0) * (1 - ratio)
    reductions[:date_idx] = 0 # No reduction before the shock
    data_target.iloc[:, 2+date_idx:] *= ratio

    # Compute the weights of the other countries
    others_weights = data_others.iloc[:, 2:] / data_others.iloc[:, 2:].sum(axis=0)
    data_others.iloc[:, 2:] += reductions * others_weights

    # Update the original data
    data[data["ExporterCountry"] == target] = data_target
    data[(data["ExporterCountry"] != target) & (data["ImporterCountry"] != target)] = data_others

    return data

def get_graph_dict(data) -> dict:
    """
    Create a dictionary of NetworkX graphs from the given data.

    Parameters:
    data (pandas.DataFrame): The input data containing information about the edges and weights.

    Returns:
    dict: A dictionary where the keys are dates and the values are NetworkX graphs.

    """
    dates = data.columns[2:].to_list()
    G_dict = {}

    for date in dates:
        slice = data.loc[:, ['ExporterCountry', 'ImporterCountry', date]]
        slice.columns = ['ExporterCountry', 'ImporterCountry', 'weight']
        G = nx.from_pandas_edgelist(slice, source='ExporterCountry', target='ImporterCountry', edge_attr="weight")
        G_dict[date] = G

    return G_dict

def plot_value_tmfg(G, node_cmap = sns.color_palette("ch:s=-0.2,r=0.6,light=0.95,dark=0.25", as_cmap=True), edge_cmap = sns.color_palette("ch:start=0.2,rot=-0.3,light=0.95,dark=0.25", as_cmap=True), seed=None, ax=None) -> nx.DiGraph:
    """
    Plot the TMFG (Triadic Motif Frequency Graph) of a given graph.

    Parameters:
    - G (networkx.Graph): The input graph.
    - node_cmap (matplotlib.colors.Colormap, optional): The colormap for node colors. Defaults to sns.color_palette("ch:s=-0.2,r=0.6,light=0.95,dark=0.25", as_cmap=True).
    - edge_cmap (matplotlib.colors.Colormap, optional): The colormap for edge colors. Defaults to sns.color_palette("ch:start=0.2,rot=-0.3,light=0.95,dark=0.25", as_cmap=True).
    - seed (int, optional): The random seed for the layout. Defaults to None.
    - ax (matplotlib.axes.Axes, optional): The axes on which to plot the graph. Defaults to None.
    """
    # Apply TMFG
    model = TMFG()
    dummy_corr = nx.to_numpy_array(G, multigraph_weight=sum, weight="weight") # multigraph_weight=sum is used to get the sum of weights of multiple edges
    dummy_corr = pd.DataFrame(dummy_corr)
    cliques, seps, adj_matrix = model.fit_transform(weights=dummy_corr, output="weighted_sparse_W_matrix")

    # Obtain TMFG graph (undirected, edge weights are sum of original bi-directional edge weights)
    G_tmfg = nx.from_numpy_array(adj_matrix)
    G_tmfg = nx.relabel_nodes(G_tmfg, dict(enumerate(G.nodes())))

    # Edge weights and colors
    edge_weights = nx.get_edge_attributes(G_tmfg, "weight")
    edge_weights = np.array(list(edge_weights.values()))
    edge_colors = edge_weights / np.max(edge_weights)

    def calculate_weighted_degree_centrality(G):
        centrality = {}
        for node in G.nodes():
            in_strength = sum(
                [edata["weight"] for u, v, edata in G.in_edges(node, data=True)]
            )
            out_strength = sum(
                [edata["weight"] for u, v, edata in G.out_edges(node, data=True)]
            )
            centrality[node] = in_strength + out_strength
        return centrality

    # Node weights and colors
    node_weights = dict(G.degree(G_tmfg.nodes(), weight="weight")) # Degree is the sum of weights of edges connected to the node
    node_weights = np.array(list(node_weights.values()))
    node_colors = node_weights / np.max(node_weights)

    # Plotting
    nx.draw(
        G_tmfg,
        # Labels control
        with_labels=True,
        font_size=8,
        # font_color="black",
        bbox=dict(facecolor="white", edgecolor="None", boxstyle="round,pad=0.0", alpha = 0.3),
        # Nodes control
        node_size=100,
        node_color=node_colors,
        cmap = node_cmap,
        vmin = 0,
        vmax = 1,
        # Edges control
        edge_color=edge_colors,
        edge_cmap = edge_cmap,
        edge_vmin = 0,
        edge_vmax = 1,
        width=edge_colors + 1,
        # Layout
        pos = nx.spring_layout(G_tmfg, k=100, seed=seed),
        ax=ax
    )

    # Add colorbar for node weights, reduce distance between the other cbar
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=plt.Normalize(vmin=node_weights.min(), vmax=node_weights.max()))
    cbar1 = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, anchor=(-0.5, 0.5))
    cbar1.ax.yaxis.set_label_position('left')   # Position of the label to the left
    cbar1.ax.yaxis.set_tick_params(rotation=90) # Rotate the ticks
    cbar1.set_label("Node Weight (Total Trading Value)")

    # Add colorbar for edge weights and reduce length of the cbar
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max()))
    cbar2 = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, anchor=(0.25, 0.5))
    cbar2.ax.yaxis.set_label_position('left')
    cbar2.ax.yaxis.set_tick_params(rotation=90)
    cbar2.set_label("Edge Weight (Sum Import/Export)")

    return G_tmfg

def plot_communities(G, method=nx.algorithms.community.louvain_communities, seed=None, ax=None) -> list:
    """
    Get communities in a graph using a specified method.

    Parameters:
    - G (networkx.Graph): The input graph.
    - method (function): The community detection method to use. Default is `nx.algorithms.community.louvain_communities`.
    - seed (int): Seed for the random number generator. Default is None.
    - ax (matplotlib.axes.Axes): The axes on which to draw the graph. Default is None.

    Returns:
    - communities (list): A list of communities, where each community is represented as a list of nodes.

    """
    communities = method(G)
    communities = [list(community) for community in communities]
    print(f"{method.__name__} found {len(communities)} communities")

    node_colors = {node: i for i, community in enumerate(communities) for node in community}
    node_cmap = sns.color_palette("viridis", as_cmap=True)
    edge_cmap = sns.color_palette("ch:start=0.2,rot=-0.3,light=0.95,dark=0.25", as_cmap=True)

    # Edge weights and colors
    edge_weights = nx.get_edge_attributes(G, "weight")
    edge_weights = np.array(list(edge_weights.values()))
    edge_colors = edge_weights / np.max(edge_weights)

    nx.draw(
        G,
        # Labels control
        with_labels=True,
        font_size=8,
        # font_color="black",
        bbox=dict(facecolor="white", edgecolor="None", boxstyle="round,pad=0.0", alpha = 0.3),
        # Nodes control
        node_size=100,
        node_color=[node_colors[node] for node in G.nodes],
        cmap = node_cmap,
        # Edges control
        edge_color=edge_colors,
        edge_cmap = edge_cmap,
        edge_vmin = 0,
        edge_vmax = 1,
        width=edge_colors/2 + 1,
        # Layout
        pos = nx.spring_layout(G, k=0.1, seed=seed),
        ax=ax
    )

    # Add colorbar for edge weights and reduce length of the cbar
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max()))
    cbar2 = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, anchor=(0.25, 0.5))
    cbar2.ax.yaxis.set_label_position('left')
    cbar2.ax.yaxis.set_tick_params(rotation=90)
    cbar2.set_label("Edge Weight (Correlation)")

    return communities