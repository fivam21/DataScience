from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


def read_clean_data(category: str = "metals", resolution: str = "annual"):
    """
    Reads and cleans the data for a specific category and resolution.

    Parameters:
    - category (str): The category of the data. Default is "metals".
    - resolution (str): The resolution of the data. Default is "annual".

    Returns:
    - data (DataFrame): The cleaned data.

    """
    data = pd.read_csv(f"cleaned_data/{resolution}_{category}.csv")
    data = data[data["Concept"] == "Total Trade Real Value"]

    # Remove vague regions
    to_drop = ["World Total", "Other Europe, nes.", "Areas, not elsewhere specified"]
    filter = np.bool_(
        ~data["ImporterCountry"].isin(to_drop) & ~data["ExporterCountry"].isin(to_drop)
    )
    data = data[filter]

    # Merge countries
    # data["ExporterCountry"] = data["ExporterCountry"].replace("New Caledonia", "France")
    # data["ExporterCountry"] = data["ExporterCountry"].replace("Taiwan", "China (mainland)")
    # data["ExporterCountry"] = data["ExporterCountry"].replace("Hong Kong SAR", "China (mainland)")

    # data["ImporterCountry"] = data["ImporterCountry"].replace("New Caledonia", "France")
    # data["ImporterCountry"] = data["ImporterCountry"].replace("Taiwan", "China (mainland)")
    # data["ImporterCountry"] = data["ImporterCountry"].replace("Hong Kong SAR", "China (mainland)")

    return data


def filter_countries(data: pd.DataFrame, countries: list):
    return data[
        np.bool_(data["ExporterCountry"].isin(countries))
        & np.bool_(data["ImporterCountry"].isin(countries))
        & np.bool_(data["ExporterCountry"] != data["ImporterCountry"])
    ]


def get_time_series(
    data: pd.DataFrame,
    countries: list,
    direction: str = "import",
    commodity: str = None,
):
    """
    Retrieves time series data for specified countries, direction, and commodity from a given DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the time series data.
        countries (list): A list of country names to filter the data.
        direction (str, optional): The direction of trade. Defaults to "import".
        commodity (str, optional): The commodity to filter the data. Defaults to None. CASE SENSITIVE

    Returns:
        pd.DataFrame: The filtered time series data.

    """
    direction = direction.lower().capitalize()
    data = data[data[f"{direction}erCountry"].isin(countries)]
    if commodity:
        data = data[data["Commodity"].str.contains(commodity)]
    data = data.groupby([f"{direction}erCountry"]).sum()
    data = data.drop(
        columns=["Concept", "ExporterCountry", "Commodity", "StartDate", "EndDate"]
    )
    data = data.T
    data.index = pd.to_datetime(data.index)
    return data


def plot_tmfg(adj_matrix, data: pd.DataFrame, title: str = "", savefig: bool = False):
    """
    Plot a directed graph using the given adjacency matrix and data.

    Parameters:
    - adj_matrix (numpy.ndarray): The adjacency matrix representing the graph.
    - data (pd.DataFrame): The processed time series data used to color the nodes and edges of the graph.

    Returns:
    None
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Add node names
    node_names = data.columns
    node_names_mapping = {i: node for i, node in enumerate(node_names)}
    G = nx.relabel_nodes(G, node_names_mapping)

    # Color the nodes by the sum of values
    node_values = data.sum()
    node_values = (node_values - node_values.min()) / (
        node_values.max() - node_values.min()
    )

    # Color the edges according to the weight
    edge_colors = [G[u][v]["weight"] for u, v in G.edges]
    edge_colors = np.array(edge_colors)
    edge_colors = (edge_colors - edge_colors.min()) / (
        edge_colors.max() - edge_colors.min()
    )

    plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        arrowstyle="-",
        node_color=node_values,
        font_color="grey",
        font_size=8,
        cmap=plt.cm.Reds,
        edge_color=edge_colors,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0,
        edge_vmax=1,
    )

    plt.title(title)
    if savefig:
        plt.savefig(f"{title}.png", dpi=800, bbox_inches="tight")
    plt.show()


def build_network(trade_data):
    """Build a network from the dataframe containing the metals trade data"""
    G = nx.MultiDiGraph()
    for _, row in trade_data.iterrows():
        importer = row["ImporterCountry"]
        exporter = row["ExporterCountry"]
        # Add nodes
        G.add_node(importer)
        G.add_node(exporter)
        # Add edges for each time slice
        for date_col in trade_data.columns[
            2:
        ]:  # Assuming first two columns are 'ImporterCountry' and 'ExporterCountry'
            date = datetime.strptime(date_col, "%Y-%m-%d")
            volume = row[date_col]
            # Check if volume is not NaN or 0, then add an edge
            if pd.notna(volume) and volume != 0:
                G.add_edge(exporter, importer, date=date, weight=volume)

    return G


def get_snapshot(G, date):
    """Get a static snapshot of the network at a specific date"""
    dt = datetime.strptime(date, "%Y-%m-%d")
    timestamp = dt.timestamp()
    snapshot_edges = [
        (u, v, key)
        for u, v, key, data in G.edges(data=True, keys=True)
        if data["date"].timestamp() == timestamp
    ]
    return nx.DiGraph(G.edge_subgraph(snapshot_edges))


def compare_degree_centrality(original_G, embargoed_G):
    """Compare the degree centrality of the original and embargoed networks"""
    orig_centrality = nx.degree_centrality(original_G)
    embargoed_centrality = nx.degree_centrality(embargoed_G)
    centrality_changes = {
        node: embargoed_centrality.get(node, 0) - orig_centrality.get(node, 0)
        for node in set(orig_centrality) | set(embargoed_centrality)
    }
    return dict(
        sorted(centrality_changes.items(), key=lambda item: abs(item[1]), reverse=True)
    )


def compare_leverage_centrality(original_G, embargoed_G):
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

    orig_centrality = calculate_weighted_degree_centrality(original_G)
    embargoed_centrality = calculate_weighted_degree_centrality(embargoed_G)

    leverage_change = {
        node: embargoed_centrality.get(node, 0) - orig_centrality.get(node, 0)
        for node in orig_centrality
    }

    return sorted(leverage_change.items(), key=lambda x: x[1], reverse=True)


def compare_pagerank(original_G, embargoed_G):
    orig_pagerank = nx.pagerank(original_G)
    embargoed_pagerank = nx.pagerank(embargoed_G)

    # Calculate the change in PageRank for each node
    pagerank_change = {
        node: embargoed_pagerank.get(node, 0) - orig_pagerank.get(node, 0)
        for node in set(orig_pagerank) | set(embargoed_pagerank)
    }

    return sorted(pagerank_change.items(), key=lambda x: abs(x[1]), reverse=True)


def compare_weighted_betweenness_centrality(original_G, embargoed_G):
    """Compare the betweenness centrality weighted by the edge weights"""
    orig_centrality = nx.betweenness_centrality(original_G, weight="weight")
    embargoed_centrality = nx.betweenness_centrality(embargoed_G, weight="weight")
    centrality_changes = {
        node: embargoed_centrality.get(node, 0) - orig_centrality.get(node, 0)
        for node in set(orig_centrality) | set(embargoed_centrality)
    }
    return dict(
        sorted(centrality_changes.items(), key=lambda item: abs(item[1]), reverse=True)
    )


def compare_strength_distribution(original_G, embargoed_G):
    """Compare weighted node importance of the networks"""
    orig_strength = dict(original_G.degree(weight="weight"))
    embargoed_strength = dict(embargoed_G.degree(weight="weight"))
    strenght_changes = {
        node: embargoed_strength.get(node, 0) - orig_strength.get(node, 0)
        for node in set(orig_strength) | set(embargoed_strength)
    }
    return dict(
        sorted(strenght_changes.items(), key=lambda item: abs(item[1]), reverse=True)
    )


def compare_weighted_clustering_coefficient(original_G, embargoed_G):
    """Compare the weighted clustering coefficient of the networks"""
    orig_clustering = nx.clustering(original_G, weight="weight")
    embargoed_clustering = nx.clustering(embargoed_G, weight="weight")
    clustering_changes = {
        node: embargoed_clustering.get(node, 0) - orig_clustering.get(node, 0)
        for node in set(orig_clustering) | set(embargoed_clustering)
    }
    return dict(
        sorted(clustering_changes.items(), key=lambda item: abs(item[1]), reverse=True)
    )


def compare_flow_hierarchy(original_G, embargoed_G):
    """Compare the flow hierarchy of the networks"""
    orig_hierarchy = nx.flow_hierarchy(original_G, weight="weight")
    embargoed_hierarchy = nx.flow_hierarchy(embargoed_G, weight="weight")
    return embargoed_hierarchy - orig_hierarchy


def compare_import_export_ratios(original_G, embargoed_G):
    """Compare the import-export ratios of the nodes in the networks"""
    orig_import_export_ratios = {
        n: original_G.in_degree(n, weight="weight")
        / original_G.out_degree(n, weight="weight")
        for n in original_G.nodes()
    }
    embargoed_import_export_ratios = {
        n: embargoed_G.in_degree(n, weight="weight")
        / embargoed_G.out_degree(n, weight="weight")
        for n in embargoed_G.nodes()
    }
    import_export_ratio_changes = {
        node: embargoed_import_export_ratios.get(node, 0)
        - orig_import_export_ratios.get(node, 0)
        for node in set(orig_import_export_ratios) | set(embargoed_import_export_ratios)
    }
    return dict(
        sorted(
            import_export_ratio_changes.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    )


def compare_weight_changes(original_G, embargoed_G):
    """Returns the weights for each network, can be used for distribution analysis"""
    orig_weights = [data["weight"] for _, _, data in original_G.edges(data=True)]
    embargoed_weights = [data["weight"] for _, _, data in embargoed_G.edges(data=True)]
    return orig_weights, embargoed_weights


def compare_edge_weight_entropy(original_G, embargoed_G):
    """Higher entropy indicates a more uniform distribution where trade volumes are
    more evenly spread out among the edges, while a lower entropy suggests a more
    uneven distribution with some edges carrying significantly more weight than others
    """

    def entropy(G):
        weights = np.array([data["weight"] for _, _, data in G.edges(data=True)])
        p = weights / weights.sum()
        return -np.sum(p * np.log(p))

    orig_entropy = entropy(original_G)
    embargoed_entropy = entropy(embargoed_G)
    return embargoed_entropy - orig_entropy
