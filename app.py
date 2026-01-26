# streamlit_app_interactive.py

import streamlit as st
import json
from typing import Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
from pyvis.network import Network

# Config

SCRATCH_JSON = "web_client_implementation/triples.json"
MISTRAL_JSON = "web-client/triples_mistral.json"
LLAMA_JSON = "web-client/triples_llama.json"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Helper Functions

def load_kg(json_path: str):
    """Load KG JSON and return nodes set, edges set, and full nodes/edges data"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes: Set[str] = set(n["data"]["id"] for n in data.get("nodes", []))
    edges: Set[Tuple[str, str, str]] = set(
        (e["data"]["source"], e["data"]["label"], e["data"]["target"])
        for e in data.get("edges", [])
    )
    return nodes, edges, data.get("nodes", []), data.get("edges", [])

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

def visualize_kg(nodes_data_list, edges_data_list, labels, height=600, width=900):
    """Visualize multiple KGs in one graph with color-coded nodes"""
    color_map = {"Scratch": "skyblue", "Mistral": "red", "LLaMA": "green"}
    net = Network(height=f"{height}px", width=f"{width}px", notebook=False, directed=True)
    
    for nodes_data, edges_data, label in zip(nodes_data_list, edges_data_list, labels):
        for node in nodes_data:
            net.add_node(node["data"]["id"], label=node["data"]["label"], color=color_map[label])
        for edge in edges_data:
            data = edge["data"]
            net.add_edge(data["source"], data["target"], title=f"{data['label']} ({label})", color=color_map[label])
    
    net.toggle_physics(True)
    return net

# Streamlit UI

st.set_page_config(page_title="Interactive KG Comparison Dashboard", layout="wide")
st.title("Interactive Knowledge Graph Comparison Dashboard")

# Load KGs

kg_data = {
    "Scratch": load_kg(SCRATCH_JSON),
    "Mistral": load_kg(MISTRAL_JSON),
    "LLaMA": load_kg(LLAMA_JSON)
}

#  Interactive KG Comparison

st.sidebar.header("Select KGs for Comparison")
kg_options = list(kg_data.keys())
kg1 = st.sidebar.selectbox("KG 1:", kg_options, index=0)
kg2 = st.sidebar.selectbox("KG 2:", kg_options, index=1)

nodes_a, edges_a, nodes_data_a, edges_data_a = kg_data[kg1]
nodes_b, edges_b, nodes_data_b, edges_data_b = kg_data[kg2]

# Metrics Display

st.subheader(f"Comparison: {kg1} vs {kg2}")

node_overlap, node_overlap_ratio = compute_node_overlap(nodes_a, nodes_b)
common_edges, precision, recall, f1, accuracy, jaccard = compute_edge_overlap(edges_a, edges_b)
stats_a = basic_graph_analysis(edges_a)
stats_b = basic_graph_analysis(edges_b)

metrics_df = pd.DataFrame([
    {"Metric": "Nodes", kg1: len(nodes_a), kg2: len(nodes_b)},
    {"Metric": "Edges", kg1: len(edges_a), kg2: len(edges_b)},
    {"Metric": "Common Nodes", kg1: len(node_overlap), kg2: len(node_overlap)},
    {"Metric": "Node Overlap Ratio", kg1: node_overlap_ratio, kg2: node_overlap_ratio},
    {"Metric": "Common Edges", kg1: len(common_edges), kg2: len(common_edges)},
    {"Metric": "Edge Precision", kg1: precision, kg2: precision},
    {"Metric": "Edge Recall", kg1: recall, kg2: recall},
    {"Metric": "Edge F1", kg1: f1, kg2: f1},
    {"Metric": "Edge Accuracy", kg1: accuracy, kg2: accuracy},
    {"Metric": "Edge Jaccard", kg1: jaccard, kg2: jaccard},
    {"Metric": "Avg Degree", kg1: stats_a[4], kg2: stats_b[4]},
    {"Metric": "Density", kg1: stats_a[5], kg2: stats_b[5]},
    {"Metric": "Largest CC", kg1: stats_a[3], kg2: stats_b[3]}
])
st.dataframe(metrics_df)

#  Metrics Plots

st.subheader("Comparison Plots")
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].bar([kg1, kg2], [len(nodes_a), len(nodes_b)], color=['skyblue','salmon'])
ax[0].set_title("Node Counts")
ax[0].set_ylabel("Number of Nodes")
ax[1].bar([kg1, kg2], [len(edges_a), len(edges_b)], color=['skyblue','salmon'])
ax[1].set_title("Edge Counts")
ax[1].set_ylabel("Number of Edges")
st.pyplot(fig)

# Combined KG Visualization 

st.subheader("Knowledge Graph Visualization (Selected KGs)")
combined_net = visualize_kg(
    [nodes_data_a, nodes_data_b],
    [edges_data_a, edges_data_b],
    [kg1, kg2]
)
combined_net.save_graph(f"{PLOTS_DIR}/kg_{kg1}_{kg2}.html")
st.components.v1.html(open(f"{PLOTS_DIR}/kg_{kg1}_{kg2}.html", 'r', encoding='utf-8').read(), height=600, width=900)

#  Show 3-way Combined KG 

if st.checkbox("Show 3-way combined KG"):
    st.subheader("Combined KG: Scratch + Mistral + LLaMA")
    combined_net_all = visualize_kg(
        [kg_data["Scratch"][2], kg_data["Mistral"][2], kg_data["LLaMA"][2]],
        [kg_data["Scratch"][3], kg_data["Mistral"][3], kg_data["LLaMA"][3]],
        ["Scratch", "Mistral", "LLaMA"]
    )
    combined_net_all.save_graph(f"{PLOTS_DIR}/kg_combined_all.html")
    st.components.v1.html(open(f"{PLOTS_DIR}/kg_combined_all.html", 'r', encoding='cutf-8').read(), height=600, width=900)
