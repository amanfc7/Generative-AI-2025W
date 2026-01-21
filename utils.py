import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import requests
import fitz # PyMuPDF

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "llama3"    #"mistral:7b"      # (can be changed dynamically before calling main)
SYSTEM_PROMPT = open("prompt.txt").read()

MAX_CHUNKS = None   # write "None" to process all chunks; change to integer if needed

def get_pdfs() -> List[str]:
    PDF_DIR = Path("pdfs")   # relative path
    pdfs = sorted(str(p) for p in PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {PDF_DIR.resolve()}")
    return pdfs

@dataclass
class Triple:
    s: str
    p: str
    o: str
    source: str

def call_ollama(prompt: str) -> Dict[str, Any]:
    """
    Call the Ollama server to get triples for a chunk of text
    """
    schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "label":  {"type": "string"}
                    },
                    "required": ["source", "target", "label"]
                }
            }
        },
        "required": ["nodes", "edges"]
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()

    return json.loads(r.json()["message"]["content"])

def extract_text(pdf_path: str) -> str:
    doc = (fitz.open(pdf_path))
    pages = []
    for i in range(len(doc)):
        pages.append(doc[i].get_text("text"))
    return "\n".join(pages)

def chunk_text(text: str, max_chars: int = 5000) -> List[str]:
    # naive chunking by paragraphs; good enough for v0
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > max_chars and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks

def normalize(x: str) -> str:
    # Match scratch KG normalization: lowercase, remove punctuation, underscores
    x = x.strip().lower()
    x = re.sub(r"[^a-z0-9_ ]", "", x)  # remove punctuation
    x = re.sub(r"\s+", "_", x)
    return x

def dedupe_triples(triples: List[Triple]) -> List[Triple]:
    seen = set()
    out = []
    for t in triples:
        key = (t.s.lower(), t.p.lower(), t.o.lower())
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out
