# Setup & Run

This project extracts knowledge graphs (KGs) from PDFs using two approaches:  

1. **Scratch KG** – traditional NLP pipeline using SpaCy for entity and relation extraction.  
2. **LLM KG** – Mistral 7B model via Ollama with a guided prompt to extract entities and edges.  

We then compare the KGs using performance metrics and generate plots.

---

## 1. Install Ollama

Make sure Ollama is installed on your system.  

- **Mac:**  

brew install ollama

Windows / Linux:
Follow instructions here: Ollama Docs

Verify installation:
ollama --version

---

## 2. Pull the model and start the server

# Start Ollama API server
ollama serve

# Pull the Mistral 7B instruct model
ollama pull mistral:7b-instruct
The server runs locally at: http://localhost:11434

Required for the LLM KG extraction pipeline

## 3. Prepare Python Environment

# Create virtual environment
python -m venv venv

# Activate environment
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install SpaCy English model:
python -m spacy download en_core_web_sm

## 4. Prepare PDF Data

Place PDFs in the pdfs/ folder.

Scratch KG and LLM KG pipelines will automatically process all PDFs in this folder.

## 5. Run Python Scripts

# 5.1 Scratch KG
- python kg_implementation.py

Extracts entities and edges using SpaCy (NER + dependency parsing).

Outputs web_client_implementation/triples.json.

# 5.2 LLM KG
- python main.py

Uses Mistral 7B via Ollama guided by prompt.txt.

Outputs web-client/triples.json.

## 6. Start Web Client
cd web-client
python -m http.server 8000

Open browser: http://localhost:8000

Visualize nodes and edges in Cytoscape.js.

## 7. Evaluate & Compare KGs

- python comparisons.py

Computes metrics:

Metric A: Graph Structure

Number of nodes and edges in Scratch KG and LLM KG.

Metric B: Node Overlap

Number of common nodes between Scratch and LLM KG.

Ratio with respect to Scratch KG.

Metric C: Edge Overlap

Precision: fraction of LLM edges present in Scratch KG.

Recall: fraction of Scratch KG edges found by LLM.

F1 Score: harmonic mean of precision and recall.

Edge Accuracy: intersection over union of edges.

and Generates plots


## Methods: 

# Scratch KG

Extracts entities using SpaCy NER and proper nouns.

Extracts relations via dependency parsing (verbs, prepositions, conjunctions).

No hardcoding of relation types; purely data-driven.

Baseline KG for comparison.

# LLM KG (Mistral 7B)

Uses Ollama to query the LLM with prompt.txt specifying entities and relation types.

Returns JSON directly with nodes and edges.

Captures abstract relationships that Scratch KG may miss.

# Evaluation Metrics

Compare structure, node overlap, and edge overlap.

Use F1 score for edges, node overlap ratio, and graph sizes.

Generate plots for visual comparison.