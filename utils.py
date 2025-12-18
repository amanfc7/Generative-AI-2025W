import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import requests
import fitz # PyMuPDF

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b-instruct"  # change to what you pulled in ollama
SYSTEM_PROMPT = open("prompt.txt").read()

MAX_CHUNKS = 1   # 👈 DEBUG: process only the first chunk

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
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "format": schema,          # 👈 key point
        "stream": False,
        "options": {"temperature": 0.1},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()

    # Ollama guarantees valid JSON here
    return json.loads(r.json()["response"])

def extract_text(pdf_path: str) -> str:
    doc = (fitz.open(pdf_path))
    pages = []
    for i in range(len(doc)):
        pages.append(doc[i].get_text("text"))
    return "\n".join(pages)

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
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
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
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