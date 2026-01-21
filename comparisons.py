import json
from typing import Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd

# Config:

SCRATCH_JSON = "web_client_implementation/triples.json"  # baseline KG
MISTRAL_JSON = "web-client/triples_mistral.json"        # Mistral KG
LLAMA_JSON = "web-client/triples_llama.json"            # LLaMA KG
PLOTS_DIR = "plots"

# Helper functions:

def load_kg(json_path: str):
    """Load KG JSON and return nodes set and edges set"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes: Set[str] = set(n["data"]["id"] for n in data.get("nodes", []))
    edges: Set[Tuple[str, str, str]] = set(
        (e["data"]["source"], e["data"]["label"], e["data"]["target"])
        for e in data.get("edges", [])
    )
    return nodes, edges

def compute_node_overlap(nodes_a, nodes_b):
    overlap = nodes_a & nodes_b
    overlap_ratio = len(overlap) / len(nodes_a) if nodes_a else 0
    return overlap, overlap_ratio

def compute_edge_overlap(edges_a, edges_b):
    common_edges = edges_a & edges_b
    precision = len(common_edges) / len(edges_b) if edges_b else 0
    recall = len(common_edges) / len(edges_a) if edges_a else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    union_edges = edges_a | edges_b
    accuracy = len(common_edges) / len(union_edges) if union_edges else 0
    jaccard = len(common_edges) / len(union_edges) if union_edges else 0
    return common_edges, precision, recall, f1, accuracy, jaccard

def basic_graph_analysis(edges: Set[Tuple[str,str,str]]):
    """Return simple NetworkX graph stats"""
    G = nx.DiGraph()
    for s, p, o in edges:
        G.add_edge(s, o, label=p)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_components = nx.number_weakly_connected_components(G)
    largest_cc = max((len(c) for c in nx.weakly_connected_components(G)), default=0)
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes else 0
    density = nx.density(G)
    return num_nodes, num_edges, num_components, largest_cc, avg_degree, density

def analyze_and_plot(json_a, json_b, label_a, label_b):
    """Run comparison between two KG JSONs (pairwise)"""
    nodes_a, edges_a = load_kg(json_a)
    nodes_b, edges_b = load_kg(json_b)

    print(f"\n--- Comparison: {label_a} vs {label_b} ---")
    print(f"Metric A: Graph Sizes")
    print(f"{label_a} KG: {len(nodes_a)} nodes, {len(edges_a)} edges")
    print(f"{label_b} KG: {len(nodes_b)} nodes, {len(edges_b)} edges")

    # Node overlap
    node_overlap, node_overlap_ratio = compute_node_overlap(nodes_a, nodes_b)
    print("\nMetric B: Node Overlap")
    print(f"Common nodes: {len(node_overlap)}")
    print(f"Overlap ratio (w.r.t {label_a}): {node_overlap_ratio:.2f}")

    # Edge overlap
    common_edges, precision, recall, f1, accuracy, jaccard = compute_edge_overlap(edges_a, edges_b)
    print("\nMetric C: Edge Overlap")
    print(f"Common edges: {len(common_edges)}")
    print(f"Precision ({label_b} w.r.t {label_a}): {precision:.2f}")
    print(f"Recall ({label_b} w.r.t {label_a}): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Edge Accuracy (intersection/union): {accuracy:.2f}")
    print(f"Edge Jaccard Similarity: {jaccard:.2f}")

    # NetworkX analysis
    stats_a = basic_graph_analysis(edges_a)
    stats_b = basic_graph_analysis(edges_b)
    print("\nNetworkX Graph Stats:")
    print(f"{label_a} - Nodes: {stats_a[0]}, Edges: {stats_a[1]}, "
          f"Weakly connected components: {stats_a[2]}, Largest CC: {stats_a[3]}, "
          f"Avg degree: {stats_a[4]:.2f}, Density: {stats_a[5]:.4f}")
    print(f"{label_b} - Nodes: {stats_b[0]}, Edges: {stats_b[1]}, "
          f"Weakly connected components: {stats_b[2]}, Largest CC: {stats_b[3]}, "
          f"Avg degree: {stats_b[4]:.2f}, Density: {stats_b[5]:.4f}")

    # Plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.bar([label_a, label_b], [len(nodes_a), len(nodes_b)], color=['skyblue', 'salmon'])
    plt.title(f"Node Count Comparison: {label_a} vs {label_b}")
    plt.ylabel("Number of Nodes")
    plt.savefig(os.path.join(PLOTS_DIR, f"nodes_comparison_{label_a}_{label_b}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar([label_a, label_b], [len(edges_a), len(edges_b)], color=['skyblue', 'salmon'])
    plt.title(f"Edge Count Comparison: {label_a} vs {label_b}")
    plt.ylabel("Number of Edges")
    plt.savefig(os.path.join(PLOTS_DIR, f"edges_comparison_{label_a}_{label_b}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(["Overlap"], [node_overlap_ratio], color='green')
    plt.title(f"Node Overlap Ratio: {label_a} vs {label_b}")
    plt.ylim(0,1)
    plt.savefig(os.path.join(PLOTS_DIR, f"node_overlap_ratio_{label_a}_{label_b}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(["Precision", "Recall", "F1"], [precision, recall, f1], color=['gold', 'orange', 'green'])
    plt.title(f"Edge Metrics: {label_a} vs {label_b}")
    plt.ylim(0,1)
    plt.savefig(os.path.join(PLOTS_DIR, f"edge_metrics_{label_a}_{label_b}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(["Edge Accuracy"], [accuracy], color='purple')
    plt.title(f"Edge Accuracy: {label_a} vs {label_b}")
    plt.ylim(0,1)
    plt.savefig(os.path.join(PLOTS_DIR, f"edge_accuracy_{label_a}_{label_b}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(["Edge Jaccard"], [jaccard], color='teal')
    plt.title(f"Edge Jaccard Similarity: {label_a} vs {label_b}")
    plt.ylim(0,1)
    plt.savefig(os.path.join(PLOTS_DIR, f"edge_jaccard_{label_a}_{label_b}.png"))
    plt.close()

    # Save metrics summary
    metrics_summary = pd.DataFrame([{
        "Metric": f"{label_a} Nodes", "Value": len(nodes_a)
    }, {
        "Metric": f"{label_b} Nodes", "Value": len(nodes_b)
    }, {
        "Metric": f"{label_a} Edges", "Value": len(edges_a)
    }, {
        "Metric": f"{label_b} Edges", "Value": len(edges_b)
    }, {
        "Metric": "Common Nodes", "Value": len(node_overlap)
    }, {
        "Metric": "Node Overlap Ratio", "Value": node_overlap_ratio
    }, {
        "Metric": "Common Edges", "Value": len(common_edges)
    }, {
        "Metric": "Edge Precision", "Value": precision
    }, {
        "Metric": "Edge Recall", "Value": recall
    }, {
        "Metric": "Edge F1", "Value": f1
    }, {
        "Metric": "Edge Accuracy", "Value": accuracy
    }, {
        "Metric": "Edge Jaccard Similarity", "Value": jaccard
    }, {
        "Metric": f"{label_a} Avg Degree", "Value": stats_a[4]
    }, {
        "Metric": f"{label_b} Avg Degree", "Value": stats_b[4]
    }, {
        "Metric": f"{label_a} Density", "Value": stats_a[5]
    }, {
        "Metric": f"{label_b} Density", "Value": stats_b[5]
    }, {
        "Metric": f"{label_a} Largest CC", "Value": stats_a[3]
    }, {
        "Metric": f"{label_b} Largest CC", "Value": stats_b[3]
    }])
    metrics_summary.to_csv(os.path.join(PLOTS_DIR, f"metrics_summary_{label_a}_{label_b}.csv"), index=False)
    print(f"\nPlots and metrics CSV for {label_a} vs {label_b} saved to {PLOTS_DIR}/")

def analyze_all_three(scratch_json, mistral_json, llama_json):
    """Compare all three KGs together"""
    scratch_nodes, scratch_edges = load_kg(scratch_json)
    mistral_nodes, mistral_edges = load_kg(mistral_json)
    llama_nodes, llama_edges = load_kg(llama_json)

    # Node overlaps
    common_nodes_all = scratch_nodes & mistral_nodes & llama_nodes
    unique_scratch = scratch_nodes - (mistral_nodes | llama_nodes)
    unique_mistral = mistral_nodes - (scratch_nodes | llama_nodes)
    unique_llama = llama_nodes - (scratch_nodes | mistral_nodes)

    # Edge overlaps
    common_edges_all = scratch_edges & mistral_edges & llama_edges
    unique_scratch_edges = scratch_edges - (mistral_edges | llama_edges)
    unique_mistral_edges = mistral_edges - (scratch_edges | llama_edges)
    unique_llama_edges = llama_edges - (scratch_edges | mistral_edges)

    # Plots
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Nodes bar
    plt.figure(figsize=(6,4))
    plt.bar(["Scratch", "Mistral", "LLaMA"], [len(scratch_nodes), len(mistral_nodes), len(llama_nodes)],
            color=['skyblue','salmon','orange'])
    plt.title("Node Counts: All 3 KGs")
    plt.ylabel("Number of Nodes")
    plt.savefig(os.path.join(PLOTS_DIR, "nodes_all_three.png"))
    plt.close()

    # Edges bar
    plt.figure(figsize=(6,4))
    plt.bar(["Scratch", "Mistral", "LLaMA"], [len(scratch_edges), len(mistral_edges), len(llama_edges)],
            color=['skyblue','salmon','orange'])
    plt.title("Edge Counts: All 3 KGs")
    plt.ylabel("Number of Edges")
    plt.savefig(os.path.join(PLOTS_DIR, "edges_all_three.png"))
    plt.close()

    # Node overlap
    plt.figure(figsize=(6,4))
    plt.bar(["Common Nodes", "Unique Scratch", "Unique Mistral", "Unique LLaMA"],
            [len(common_nodes_all), len(unique_scratch), len(unique_mistral), len(unique_llama)],
            color=['green','skyblue','salmon','orange'])
    plt.title("Node Overlap: All 3 KGs")
    plt.ylabel("Number of Nodes")
    plt.savefig(os.path.join(PLOTS_DIR, "node_overlap_all_three.png"))
    plt.close()

    # Edge overlap
    plt.figure(figsize=(6,4))
    plt.bar(["Common Edges", "Unique Scratch", "Unique Mistral", "Unique LLaMA"],
            [len(common_edges_all), len(unique_scratch_edges), len(unique_mistral_edges), len(unique_llama_edges)],
            color=['green','skyblue','salmon','orange'])
    plt.title("Edge Overlap: All 3 KGs")
    plt.ylabel("Number of Edges")
    plt.savefig(os.path.join(PLOTS_DIR, "edge_overlap_all_three.png"))
    plt.close()

    # CSV summary
    all_three_metrics = pd.DataFrame([{
        "Metric": "Scratch Nodes", "Value": len(scratch_nodes)
    }, {
        "Metric": "Mistral Nodes", "Value": len(mistral_nodes)
    }, {
        "Metric": "LLaMA Nodes", "Value": len(llama_nodes)
    }, {
        "Metric": "Scratch Edges", "Value": len(scratch_edges)
    }, {
        "Metric": "Mistral Edges", "Value": len(mistral_edges)
    }, {
        "Metric": "LLaMA Edges", "Value": len(llama_edges)
    }, {
        "Metric": "Common Nodes (All 3)", "Value": len(common_nodes_all)
    }, {
        "Metric": "Unique Scratch Nodes", "Value": len(unique_scratch)
    }, {
        "Metric": "Unique Mistral Nodes", "Value": len(unique_mistral)
    }, {
        "Metric": "Unique LLaMA Nodes", "Value": len(unique_llama)
    }, {
        "Metric": "Common Edges (All 3)", "Value": len(common_edges_all)
    }, {
        "Metric": "Unique Scratch Edges", "Value": len(unique_scratch_edges)
    }, {
        "Metric": "Unique Mistral Edges", "Value": len(unique_mistral_edges)
    }, {
        "Metric": "Unique LLaMA Edges", "Value": len(unique_llama_edges)
    }])
    all_three_metrics.to_csv(os.path.join(PLOTS_DIR, "metrics_all_three.csv"), index=False)
    print("\nPlots and CSV for all three KGs saved to", PLOTS_DIR)


# Run comparisons
if __name__ == "__main__":
    # Pairwise
    analyze_and_plot(SCRATCH_JSON, MISTRAL_JSON, "Scratch", "Mistral")
    analyze_and_plot(SCRATCH_JSON, LLAMA_JSON, "Scratch", "LLaMA")
    analyze_and_plot(MISTRAL_JSON, LLAMA_JSON, "Mistral", "LLaMA")

    # All three
    analyze_all_three(SCRATCH_JSON, MISTRAL_JSON, LLAMA_JSON)
