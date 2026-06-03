"""
Micro-Level Social Network Analysis
-------------------------------------
Analyzes a small social network using:
- Degree Centrality       : Who has the most connections?
- Clustering Coefficient  : How tightly knit are a user's friends?
- Shortest Path           : Minimum hops between two users
- Community Detection     : Which groups naturally form?
- Friend Recommendation   : Who should each user connect with next?
"""

import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def load_graph(filepath):
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return nx.from_pandas_edgelist(df, source="User", target="Friend")


def degree_centrality(G):
    return sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)


def clustering_coefficients(G):
    return sorted(nx.clustering(G).items(), key=lambda x: x[1], reverse=True)


def shortest_paths(G):
    results = []
    nodes = sorted(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if nx.has_path(G, u, v):
                path = nx.shortest_path(G, u, v)
                results.append((u, v, len(path) - 1, " -> ".join(path)))
    return results


def detect_communities(G):
    communities = greedy_modularity_communities(G)
    return [sorted(c) for c in communities]


def recommend_friends(G, user, top_n=2):
    if user not in G:
        return []
    direct = set(G.neighbors(user))
    candidates = {}
    for friend in direct:
        for fof in G.neighbors(friend):
            if fof != user and fof not in direct:
                mutual = len(direct & set(G.neighbors(fof)))
                candidates[fof] = mutual
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]


def generate_report(G, filepath="report.txt"):
    lines = []

    lines.append("=" * 60)
    lines.append("   MICRO-LEVEL SOCIAL NETWORK ANALYSIS REPORT")
    lines.append("=" * 60)

    # Basic stats
    lines.append("\n[1] NETWORK OVERVIEW")
    lines.append(f"  Total Users (Nodes) : {G.number_of_nodes()}")
    lines.append(f"  Total Friendships   : {G.number_of_edges()}")
    lines.append(f"  Network Density     : {nx.density(G):.4f}")
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    lines.append(f"  Average Degree      : {avg_deg:.2f}")

    # Degree centrality
    lines.append("\n[2] DEGREE CENTRALITY (Most Connected Users)")
    for user, score in degree_centrality(G):
        lines.append(f"  {user:<10} {score:.4f}")

    # Clustering
    lines.append("\n[3] CLUSTERING COEFFICIENT (Friend Group Tightness)")
    for user, score in clustering_coefficients(G):
        lines.append(f"  {user:<10} {score:.4f}")

    # Shortest paths
    lines.append("\n[4] SHORTEST PATHS BETWEEN ALL USER PAIRS")
    for u, v, hops, path in shortest_paths(G):
        lines.append(f"  {u} <-> {v} : {hops} hop(s)  [{path}]")

    # Communities
    lines.append("\n[5] COMMUNITY DETECTION (Natural Groups)")
    for i, community in enumerate(detect_communities(G), 1):
        lines.append(f"  Group {i}: {', '.join(community)}")

    # Friend recommendations
    lines.append("\n[6] FRIEND RECOMMENDATIONS")
    for user in sorted(G.nodes()):
        recs = recommend_friends(G, user)
        if recs:
            rec_str = ", ".join([f"{c} ({m} mutual)" for c, m in recs])
            lines.append(f"  {user:<10} -> {rec_str}")
        else:
            lines.append(f"  {user:<10} -> No recommendations")

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    with open(filepath, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    G = load_graph("network.csv")
    generate_report(G)
