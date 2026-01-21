import json
from typing import List, Dict
from tqdm import tqdm
from utils import extract_text, chunk_text, call_ollama, Triple, normalize, dedupe_triples, get_pdfs, MAX_CHUNKS, MODEL


def main(pdf_paths: List[str], out_path: str = "web-client/triples.json"):
    """
    Main function to extract triples from PDFs using a specified LLM.
    Saves the results in a JSON file (Cytoscape-compatible).
    """
    prompt = open("prompt.txt").read()
    all_triples: List[Triple] = []
    all_entities: Dict[str, str] = {}  # id -> type

    for pdf in pdf_paths:
        text = extract_text(pdf)
        chunks = chunk_text(text)

        # Process all chunks if MAX_CHUNKS is None
        for i, chunk in enumerate(tqdm(chunks[:MAX_CHUNKS] if MAX_CHUNKS else chunks,
                                       desc=f"LLM extracting: {pdf}", unit="chunk")):
            user_prompt = f"{prompt}\n\nTEXT:\n{chunk}\n"
            data = call_ollama(user_prompt)

            # Add nodes
            for n in data.get("nodes", []):
                eid = normalize(n)
                if eid:
                    all_entities.setdefault(eid, "Entity")

            # Add edges without filtering
            for e in data.get("edges", []):
                s = normalize(e.get("source", ""))
                p = normalize(e.get("label", ""))
                o = normalize(e.get("target", ""))
                if s and p and o:
                    all_triples.append(Triple(s=s, p=p, o=o, source=pdf))

        all_triples = dedupe_triples(all_triples)

    # convert to a browser-friendly Cytoscape format
    nodes = []
    node_ids = set()
    for eid, etype in all_entities.items():
        node_ids.add(eid)
        nodes.append({"data": {"id": eid, "label": eid, "type": etype}})

    # ensure nodes for any triple endpoints (even if entities list missed them)
    for t in all_triples:
        for eid in (t.s, t.o):
            if eid not in node_ids:
                node_ids.add(eid)
                nodes.append({"data": {"id": eid, "label": eid, "type": "Entity"}})

    edges = []
    for i, t in enumerate(all_triples):
        edges.append({
            "data": {
                "id": f"e{i}",
                "source": t.s,
                "target": t.o,
                "label": t.p,
                "paper": t.source
            }
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} with {len(nodes)} nodes and {len(edges)} edges.")


if __name__ == "__main__":
    pdfs = get_pdfs()
    models = ["mistral:7b", "llama3"]  # List of models to run

    for model_name in models:
        MODEL = model_name  # dynamically set the model before calling main
        out_file = f"web-client/triples_{model_name.split(':')[0]}.json"
        print(f"\nProcessing PDFs with {model_name}...")
        main(pdfs, out_path=out_file)
        print(f"Done with {model_name}, saved to {out_file}")
