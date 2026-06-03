from flask import Flask, render_template, request
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

app = Flask(__name__)


def build_graph(connections):
    G = nx.Graph()
    for line in connections.strip().split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            owner = parts[0]
            for friend in parts[1:]:
                G.add_edge(owner, friend)
    return G


def analyze(G):
    overview = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
    }

    centrality = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)
    centrality = [(u, round(s, 4)) for u, s in centrality]

    clustering = sorted(nx.clustering(G).items(), key=lambda x: x[1], reverse=True)
    clustering = [(u, round(s, 4)) for u, s in clustering]

    nodes = sorted(G.nodes())
    paths = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if nx.has_path(G, u, v):
                path = nx.shortest_path(G, u, v)
                paths.append((u, v, len(path) - 1, " → ".join(path)))

    communities = [sorted(c) for c in greedy_modularity_communities(G)]

    recommendations = []
    for user in sorted(G.nodes()):
        direct = set(G.neighbors(user))
        candidates = {}
        for friend in direct:
            for fof in G.neighbors(friend):
                if fof != user and fof not in direct:
                    mutual = len(direct & set(G.neighbors(fof)))
                    candidates[fof] = mutual
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:2]
        recommendations.append((user, ranked if ranked else [("No recommendation", "-")]))

    return overview, centrality, clustering, paths, communities, recommendations


def generate_charts(G):
    os.makedirs("static/images", exist_ok=True)

    # Network graph
    communities = list(greedy_modularity_communities(G))
    color_list = ["#f953c6", "#4776e6", "#38ef7d", "#ffd200", "#ff6b6b"]
    node_colors = {}
    for i, community in enumerate(communities):
        for node in community:
            node_colors[node] = color_list[i % len(color_list)]

    colors = [node_colors[n] for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
    ax = plt.gca()
    ax.set_facecolor("#1a1a2e")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=900, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#555", width=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", font_weight="bold", ax=ax)
    legend_patches = [
        mpatches.Patch(color=color_list[i], label=f"Group {i+1}: {', '.join(sorted(c))}")
        for i, c in enumerate(communities)
    ]
    plt.legend(handles=legend_patches, loc="upper left", fontsize=8,
               facecolor="#16162a", labelcolor="white", edgecolor="#333")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("static/images/network_graph.png", dpi=150, facecolor="#1a1a2e")
    plt.close()

    # Degree centrality bar
    centrality = nx.degree_centrality(G)
    users = sorted(centrality, key=centrality.get, reverse=True)
    scores = [centrality[u] for u in users]

    plt.figure(figsize=(14, 6), facecolor="#1a1a2e")
    ax = plt.gca()
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(users, scores, color=["#f953c6", "#4776e6", "#38ef7d", "#ffd200", "#ff6b6b",
                                         "#a29bfe", "#fd79a8", "#00cec9"][:len(users)], edgecolor="#1a1a2e")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.2f}", ha="center", va="bottom", fontsize=9, color="white")
    ax.set_title("Degree Centrality", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("User", color="#aaa")
    ax.set_ylabel("Score", color="#aaa")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    plt.tight_layout()
    plt.savefig("static/images/degree_centrality.png", dpi=150, facecolor="#1a1a2e")
    plt.close()

    # Clustering bar
    clustering = nx.clustering(G)
    users_c = sorted(clustering, key=clustering.get, reverse=True)
    scores_c = [clustering[u] for u in users_c]

    plt.figure(figsize=(14, 6), facecolor="#1a1a2e")
    ax = plt.gca()
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(users_c, scores_c, color=["#8e54e9", "#38ef7d", "#f953c6", "#ffd200", "#4776e6",
                                              "#fd79a8", "#00cec9", "#a29bfe"][:len(users_c)], edgecolor="#1a1a2e")
    for bar, score in zip(bars, scores_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.2f}", ha="center", va="bottom", fontsize=9, color="white")
    ax.set_title("Clustering Coefficient", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("User", color="#aaa")
    ax.set_ylabel("Score", color="#aaa")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    plt.ylim(0, max(scores_c) + 0.2 if max(scores_c) > 0 else 0.5)
    plt.tight_layout()
    plt.savefig("static/images/clustering.png", dpi=150, facecolor="#1a1a2e")
    plt.close()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/analyze", methods=["POST"])
def analyze_network():
    connections = request.form.get("connections", "")
    if not connections.strip():
        return render_template("index.html", result=None, error="Please enter at least one connection.")
    try:
        G = build_graph(connections)
        if G.number_of_nodes() == 0:
            return render_template("index.html", result=None, error="Invalid input format. Use: Name1, Name2")
        overview, centrality, clustering, paths, communities, recommendations = analyze(G)
        generate_charts(G)
        return render_template("index.html",
                               result=True,
                               connections=connections,
                               overview=overview,
                               centrality=centrality,
                               clustering=clustering,
                               paths=paths,
                               communities=communities,
                               recommendations=recommendations)
    except Exception as e:
        return render_template("index.html", result=None, error=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
