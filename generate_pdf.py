"""
Generates a formatted PDF report for the Micro-Level Social Network Analysis.
"""

import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)


def load_graph():
    df = pd.read_csv("network.csv")
    return nx.from_pandas_edgelist(df, source="User", target="Friend")


def build_pdf():
    G = load_graph()
    doc = SimpleDocTemplate(
        "report.pdf",
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", fontSize=18, fontName="Helvetica-Bold",
                                  spaceAfter=6, textColor=colors.HexColor("#2C3E50"), alignment=1)
    subtitle_style = ParagraphStyle("subtitle", fontSize=11, fontName="Helvetica",
                                     spaceAfter=12, textColor=colors.grey, alignment=1)
    section_style = ParagraphStyle("section", fontSize=13, fontName="Helvetica-Bold",
                                    spaceAfter=6, textColor=colors.HexColor("#2980B9"),
                                    spaceBefore=14)
    body_style = ParagraphStyle("body", fontSize=10, fontName="Helvetica",
                                 spaceAfter=4, leading=14)

    story = []

    # Title
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Micro-Level Social Network Analysis", title_style))
    story.append(Paragraph("Friend Recommendation System — Project Report", subtitle_style))
    story.append(Spacer(1, 0.5 * cm))

    # Section 1: Overview
    story.append(Paragraph("1. Network Overview", section_style))
    density = nx.density(G)
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    overview_data = [
        ["Metric", "Value"],
        ["Total Users (Nodes)", str(G.number_of_nodes())],
        ["Total Friendships (Edges)", str(G.number_of_edges())],
        ["Network Density", f"{density:.4f}"],
        ["Average Degree", f"{avg_deg:.2f}"],
    ]
    story.append(_table(overview_data))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "The network consists of 8 users connected by 11 friendship edges. "
        "A density of 0.39 indicates a moderately connected network where users "
        "are neither fully isolated nor fully interconnected.",
        body_style
    ))

    # Section 2: Degree Centrality
    story.append(Paragraph("2. Degree Centrality", section_style))
    story.append(Paragraph(
        "Degree centrality measures how many direct connections each user has. "
        "A higher score means the user is more influential in the network.",
        body_style
    ))
    centrality = nx.degree_centrality(G)
    cent_data = [["User", "Degree Centrality"]] + [
        [u, f"{s:.4f}"] for u, s in sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    ]
    story.append(_table(cent_data))

    # Section 3: Clustering Coefficient
    story.append(Paragraph("3. Clustering Coefficient", section_style))
    story.append(Paragraph(
        "The clustering coefficient measures how tightly knit a user's friends are. "
        "A score of 1.0 means all of a user's friends are also friends with each other.",
        body_style
    ))
    clustering = nx.clustering(G)
    clust_data = [["User", "Clustering Coefficient"]] + [
        [u, f"{s:.4f}"] for u, s in sorted(clustering.items(), key=lambda x: x[1], reverse=True)
    ]
    story.append(_table(clust_data))

    # Section 4: Shortest Paths
    story.append(Paragraph("4. Shortest Paths Between Users", section_style))
    story.append(Paragraph(
        "Shortest path shows the minimum number of hops needed to connect two users, "
        "reflecting how closely related they are in the network.",
        body_style
    ))
    nodes = sorted(G.nodes())
    path_data = [["User A", "User B", "Hops", "Path"]]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if nx.has_path(G, u, v):
                path = nx.shortest_path(G, u, v)
                path_data.append([u, v, str(len(path) - 1), " → ".join(path)])
    story.append(_table(path_data, col_widths=[2.5 * cm, 2.5 * cm, 2 * cm, 9 * cm]))

    # Section 5: Communities
    story.append(Paragraph("5. Community Detection", section_style))
    story.append(Paragraph(
        "Using the Greedy Modularity algorithm, users are grouped into natural communities "
        "based on the density of their connections.",
        body_style
    ))
    communities = list(greedy_modularity_communities(G))
    comm_data = [["Group", "Members"]] + [
        [f"Group {i+1}", ", ".join(sorted(c))] for i, c in enumerate(communities)
    ]
    story.append(_table(comm_data))

    # Section 6: Friend Recommendations
    story.append(Paragraph("6. Friend Recommendations", section_style))
    story.append(Paragraph(
        "Recommendations are based on mutual friend count. Users with more mutual friends "
        "are ranked higher as potential connections.",
        body_style
    ))
    rec_data = [["User", "Recommended Friend", "Mutual Friends"]]
    for user in sorted(G.nodes()):
        direct = set(G.neighbors(user))
        candidates = {}
        for friend in direct:
            for fof in G.neighbors(friend):
                if fof != user and fof not in direct:
                    mutual = len(direct & set(G.neighbors(fof)))
                    candidates[fof] = mutual
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:2]
        if ranked:
            for candidate, mutual in ranked:
                rec_data.append([user, candidate, str(mutual)])
        else:
            rec_data.append([user, "No recommendation", "-"])
    story.append(_table(rec_data))

    # Page break before visualizations
    story.append(PageBreak())
    story.append(Paragraph("7. Visualizations", section_style))

    # Network graph
    story.append(Paragraph("Social Network Graph (Communities Colored):", body_style))
    story.append(Image("network_graph.png", width=14 * cm, height=10 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # Degree centrality chart
    story.append(Paragraph("Degree Centrality per User:", body_style))
    story.append(Image("degree_centrality.png", width=14 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # Clustering chart
    story.append(Paragraph("Clustering Coefficient per User:", body_style))
    story.append(Image("clustering.png", width=14 * cm, height=8 * cm))

    # Conclusion
    story.append(PageBreak())
    story.append(Paragraph("8. Conclusion", section_style))
    story.append(Paragraph(
        "This micro-level social network analysis examined a small network of 8 users. "
        "Key findings include: Alice, Bob, Charlie, David, Eve, and Frank are the most central users "
        "with equal degree centrality scores. Alice, Bob, and Charlie form tighter friend clusters "
        "compared to others. Community detection revealed two natural groups — one centered around "
        "Alice/Bob/Charlie and another around David/Eve/Frank/Grace/Henry. "
        "The friend recommendation system successfully identified meaningful connections "
        "based on mutual friend overlap, which can scale to larger real-world networks.",
        body_style
    ))

    doc.build(story)
    print("Report saved as: report.pdf")


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980B9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EBF5FB")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


if __name__ == "__main__":
    build_pdf()
