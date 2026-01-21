"""
This script reads PDFs and extracts entities and
relationships relevant to scientific documents, such as models, datasets, tasks, metrics, 
and system components. It then generates triples of the form (source_entity, relation, target_entity)
and outputs a JSON file compatible with Cytoscape for visualization.

Pipeline Overview:

1. PDF Processing:
    - Read PDFs from a specified folder.
    - Extract raw text from each PDF using PyMuPDF (fitz).
    - Preprocess text: remove extra spaces, headers/footers, references, stopwords.
    - Split text into chunks for analysis.

2. Entity Extraction:
    - Uses SpaCy's named entity recognition (NER) for initial entity identification.
    - Additionally extracts proper nouns as candidate entities.
    - Normalizes entities by converting to lowercase, removing punctuation, and replacing spaces with underscores.

3. Relation Extraction:
    - Uses dependency parsing to detect verb-based relationships between entities.
    - Expanded to include:
        * nouns connected to verbs
        * conjunction handling (multiple entities connected via same verb)
        * wider preposition patterns
    - Only creates triples if both source and target are recognized entities.

4. Deduplication:
    - Removes duplicate triples to ensure a clean knowledge graph.

5. Output Generation:
    - Converts entities and triples into a JSON format compatible with Cytoscape or other KG visualization tools.
    - Saves the JSON to a specified output path.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import fitz  # PyMuPDF
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Configuration: 

PDF_DIR = Path("pdfs")  # folder containing PDFs
OUT_PATH = "web_client_implementation/triples.json"
CHUNK_SIZE = 5000  # max chars per chunk for splitting
# meaningful preposition whitelist for KG
PREPOSITION_RELATIONS = {"with", "using", "based_on", "applied_to", "for", "on", "via", "through"}

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Data Structures:

@dataclass
class Triple:
    s: str  # source entity
    p: str  # relation
    o: str  # target entity
    source: str  # source of triple (PDF)

# PDF Functions:

def get_pdfs() -> List[str]:
    """Return list of PDF file paths in PDF_DIR."""
    pdfs = sorted(str(p) for p in PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {PDF_DIR.resolve()}")
    return pdfs

def extract_text(pdf_path: str) -> str:
    """Extract and preprocess text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [doc[i].get_text("text") for i in range(len(doc))]
    text = "\n".join(pages)

    # Basic preprocessing:

    text = re.sub(r"\n\s*\n", "\n\n", text)  # collapse empty lines
    text = re.sub(r"\d+\s*\n", "\n", text)  # remove isolated numbers (page numbers)
    text = re.sub(r"References|REFERENCE[S]?", "", text, flags=re.I)  # remove reference headers
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    text = text.strip()

    # Stopword removal
    words = text.split()
    text = " ".join([w for w in words if w.lower() not in STOP_WORDS])
    return text

def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks not exceeding max_chars."""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i+max_chars])
    return chunks


# Normalization:

def normalize(x: str) -> str:
    """Normalize entity/relation strings."""
    x = x.strip().lower()
    x = re.sub(r"[^a-z0-9_ ]", "", x)  # remove punctuation
    x = re.sub(r"\s+", "_", x)  # replace spaces with underscore
    return x

def dedupe_triples(triples: List[Triple]) -> List[Triple]:
    """Remove duplicate triples."""
    seen = set()
    out = []
    for t in triples:
        key = (t.s, t.p, t.o)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

# Entity Extraction:

def extract_entities(text: str) -> List[str]:
    """Extract entities from text using SpaCy (NER + proper nouns)."""
    doc = nlp(text)
    entities = set()

    # 1. Add all recognized named entities
    for ent in doc.ents:
        entities.add(normalize(ent.text))

    # 2. include proper nouns
    for token in doc:
        if token.pos_ == "PROPN" and len(token.text) > 2:
            entities.add(normalize(token.text))

    return list(entities)

# Relation Extraction:

def extract_relations(text: str, entities: List[str]) -> List[Triple]:
    """
    Extract relations including:
    - verb-mediated relations
    - nouns connected to verbs
    - prepositional relations
    - conjunction handling
    """
    triples = []
    doc = nlp(text)

    for sent in doc.sents:
        # map token index -> entity
        token_to_entity = {token.i: normalize(token.text) for token in sent if normalize(token.text) in entities}

        # Verb-mediated + noun-object relations
        for token in sent:
            if token.pos_ == "VERB":
                # subjects
                subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass") and w.i in token_to_entity]
                # objects: direct objects + dative + noun compounds
                objects = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "dative") and w.i in token_to_entity]

                # include connected nouns (e.g., performance metric)
                for obj in list(objects):
                    for child in obj.children:
                        if child.dep_ in ("compound", "amod") and normalize(child.text) in entities:
                            objects.append(child)

                # conjunction handling
                for s in subjects:
                    s_entities = [s]
                    for conj in s.conjuncts:
                        if conj.i in token_to_entity:
                            s_entities.append(conj)
                    for o in objects:
                        o_entities = [o]
                        for conj_o in o.conjuncts:
                            if conj_o.i in token_to_entity:
                                o_entities.append(conj_o)
                        for s_ent in s_entities:
                            for o_ent in o_entities:
                                s_norm = token_to_entity[s_ent.i]
                                o_norm = token_to_entity[o_ent.i]
                                if s_norm != o_norm:
                                    triples.append(Triple(s=s_norm, p=normalize(token.lemma_), o=o_norm, source="PDF"))

        # Prepositional relations
        for token in sent:
            if token.dep_ == "prep" and token.lemma_ in PREPOSITION_RELATIONS:
                prep_objs = [w for w in token.rights if w.i in token_to_entity]
                prep_head = token.head
                if prep_head.i in token_to_entity:
                    s_norm = token_to_entity[prep_head.i]
                    for o in prep_objs:
                        o_norm = token_to_entity[o.i]
                        if s_norm != o_norm:
                            triples.append(Triple(s=s_norm, p=normalize(token.lemma_), o=o_norm, source="PDF"))

    return triples

# Main Function:

def main():
    pdf_paths = get_pdfs()
    all_triples: List[Triple] = []
    all_entities: Dict[str, str] = {} 

    for pdf in pdf_paths:
        text = extract_text(pdf)
        chunks = chunk_text(text)

        for chunk in chunks:
            entities = extract_entities(chunk)
            for e in entities:
                all_entities.setdefault(e, "Entity")

            triples = extract_relations(chunk, entities)
            all_triples.extend(triples)

    all_triples = dedupe_triples(all_triples)

    # Convert to Cytoscape JSON:

    nodes = [{"data": {"id": eid, "label": eid, "type": etype}} for eid, etype in all_entities.items()]
    node_ids = set(all_entities.keys())

    # Ensure all nodes in edges exist:

    for t in all_triples:
        for eid in (t.s, t.o):
            if eid not in node_ids:
                node_ids.add(eid)
                nodes.append({"data": {"id": eid, "label": eid, "type": "Entity"}})

    edges = [
        {"data": {"id": f"e{i}", "source": t.s, "target": t.o, "label": t.p, "paper": t.source}}
        for i, t in enumerate(all_triples)
    ]

    # Save JSON:

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH} with {len(nodes)} nodes and {len(edges)} edges.")


if __name__ == "__main__":
    main()
