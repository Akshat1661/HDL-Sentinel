"""
build_code_rag.py — Ingestion pipeline for the Verilog code-example RAG corpus.

Reads verified_dataset_14k.jsonl, extracts metadata, embeds prompts with the
same sentence-transformer used by app2.py, and stores everything in a
persistent ChromaDB at ./chroma_db_code_examples/.

Usage:
    python build_code_rag.py

    Optional flags:
        --jsonl  PATH    Path to the JSONL dataset (default: ./verified_dataset_14k.jsonl)
        --db     PATH    ChromaDB output directory  (default: ./chroma_db_code_examples)
        --model  PATH    Embedding model directory  (default: ./embedding_model)
        --batch  N       Embedding batch size       (default: 256)
        --reset          Delete and rebuild collection from scratch (default: True)

Requirements:
    pip install chromadb sentence-transformers tqdm
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# --- CLI args ---
parser = argparse.ArgumentParser()
parser.add_argument("--jsonl",  default="verified_dataset_14k.jsonl")
parser.add_argument("--db",     default="chroma_db_code_examples")
parser.add_argument("--model",  default="embedding_model")
parser.add_argument("--batch",  type=int, default=256)
parser.add_argument("--no-reset", dest="reset", action="store_false",
                    help="Skip deleting existing collection (append mode).")
parser.set_defaults(reset=True)
args = parser.parse_args()

SCRIPT_DIR    = Path(__file__).parent
JSONL_PATH    = Path(args.jsonl) if Path(args.jsonl).is_absolute() else SCRIPT_DIR / args.jsonl
DB_PATH       = Path(args.db)   if Path(args.db).is_absolute()    else SCRIPT_DIR / args.db
MODEL_PATH    = Path(args.model) if Path(args.model).is_absolute() else SCRIPT_DIR / args.model
BATCH_SIZE    = args.batch
RESET         = args.reset
COLLECTION    = "verilog_code_examples"

# ---------------------------------------------------------------------------
# 1. Validate inputs
# ---------------------------------------------------------------------------
if not JSONL_PATH.exists():
    sys.exit(f"[ERROR] Dataset not found: {JSONL_PATH}")
if not MODEL_PATH.is_dir():
    sys.exit(f"[ERROR] Embedding model not found: {MODEL_PATH}\n"
             "        Run app2.py at least once to download the model, or set --model.")

# ---------------------------------------------------------------------------
# 2. Design-type classifier (mirrors _DESIGN_TYPE_KEYWORDS in app2.py)
# ---------------------------------------------------------------------------
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "dff":        ["d flip-flop", "d-flip-flop", "dff", "d flop", "d-flop",
                   "d type flip", "dtype flip"],
    "tff":        ["t flip-flop", "t-flip-flop", "tff", "toggle flip"],
    "jkff":       ["jk flip-flop", "jk-flip-flop", "jkff", "jk flop"],
    "johnson":    ["johnson counter", "johnson"],
    "lfsr":       ["lfsr", "linear feedback shift", "linear-feedback"],
    "shift_reg":  ["shift register", "shift reg", "sipo", "siso", "piso", "pipo",
                   "serial in", "serial-in", "circular shift"],
    "counter":    ["counter", "count up", "count down", "binary counter",
                   "bcd counter", "up counter", "down counter", "mod-", "modulo",
                   "incrementer", "decrementer"],
    "mux":        ["mux", "multiplexer", "2-to-1", "4-to-1", "8-to-1",
                   "2:1", "4:1", "8:1"],
    "adder":      ["adder", "half adder", "full adder", "ripple carry",
                   "carry lookahead", "carry-lookahead"],
    "subtractor": ["subtractor"],
    "alu":        ["alu", "arithmetic logic unit"],
    "decoder":    ["decoder", "demux", "demultiplexer"],
    "encoder":    ["encoder", "priority encoder"],
    "comparator": ["comparator", "compare", "power of 2"],
    "converter":  ["converter", "gray code", "bcd-to", "binary-to",
                   "one-hot", "thermometer"],
    "gate":       ["and gate", "or gate", "nand gate", "nor gate", "xor gate",
                   "xnor gate", "not gate", "inverter", "logic gate",
                   "bitwise and", "bitwise or"],
    "fsm":        ["fsm", "state machine", "moore", "mealy", "traffic light",
                   "sequence detector", "finite state"],
    "fifo":       ["fifo", "queue"],
    "ram":        ["ram", "sram", "memory array"],
    "uart":       ["uart", "serial comm", "baud"],
}

_SEQ_OVERRIDE = [
    "flip-flop", "flop", "latch", "register", "counter", "fsm", "state machine",
    "synchronous", "clocked", "sequential", "pipeline", "fifo", "ram", "uart",
]

def _classify_type(prompt: str) -> str:
    p = prompt.lower()
    for dtype, kws in _TYPE_KEYWORDS.items():
        if any(kw in p for kw in kws):
            return dtype
    # fallback: sequential vs combinational
    if any(kw in p for kw in _SEQ_OVERRIDE):
        return "sequential"
    return "combinational"


# ---------------------------------------------------------------------------
# 3. Metadata extractor
# ---------------------------------------------------------------------------
def extract_metadata(entry: dict) -> dict:
    prompt      = entry.get("prompt", "")
    design_code = entry.get("design_code", "")
    p_lower  = prompt.lower()
    dc_lower = design_code.lower()

    # Sequential: check design_code (more reliable than prompt keyword matching)
    is_seq = int("posedge clk" in dc_lower or "negedge clk" in dc_lower)

    # Reset type from prompt
    if "asynchronous" in p_lower or "async" in p_lower:
        reset_type = "async"
    elif "synchronous" in p_lower or "sync" in p_lower:
        reset_type = "sync"
    else:
        reset_type = "none"

    # Reset polarity from prompt
    if "active-high" in p_lower or "active high" in p_lower:
        reset_polarity = "active_high"
    elif "active-low" in p_lower or "active low" in p_lower:
        reset_polarity = "active_low"
    else:
        reset_polarity = "unknown"

    # Bit width — first "N-bit" or "N bit" in prompt
    m = re.search(r"(\d+)\s*-?\s*bit", p_lower)
    bit_width = int(m.group(1)) if m else 0

    # Parameterized?
    is_param = int("parameter" in dc_lower or "parameterized" in p_lower)

    # ChromaDB only accepts str / int / float in metadata — booleans must be int
    return {
        "design_type":    _classify_type(prompt),
        "is_sequential":  is_seq,
        "reset_type":     reset_type,
        "reset_polarity": reset_polarity,
        "bit_width":      bit_width,
        "is_parameterized": is_param,
        "has_testbench":  1,          # all entries in this dataset have a testbench
        "prompt_len":     len(prompt),
    }


# ---------------------------------------------------------------------------
# 4. Document builder
# ---------------------------------------------------------------------------
def build_document(entry: dict) -> str:
    """
    What we store as the retrieval document.

    Format:
        // PROMPT: <prompt>
        //
        <design_code>

    Rationale:
    - The model uses retrieved documents as reference code examples.
    - Storing only design_code (not testbench) keeps documents short (~300-900 chars)
      so 2-3 examples fit within the 3000-char context budget.
    - The prompt comment header tells the model what design this example implements,
      which helps it transfer the pattern to the user's slightly different spec.
    - testbench_code is NOT stored in the document: it would double chunk size and
      the TDD pipeline generates its own testbench independently in Pass 1.
    """
    prompt      = entry.get("prompt", "").strip()
    design_code = entry.get("design_code", "").strip()
    return f"// PROMPT: {prompt}\n//\n{design_code}"


# ---------------------------------------------------------------------------
# 5. Load dataset
# ---------------------------------------------------------------------------
print(f"[1/5] Loading dataset from {JSONL_PATH} ...")
entries = []
with open(JSONL_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed line: {e}")

print(f"      Loaded {len(entries):,} entries.")


# ---------------------------------------------------------------------------
# 6. Build documents, metadata, IDs
# ---------------------------------------------------------------------------
print("[2/5] Extracting metadata and building documents ...")
documents = []
metadatas = []
ids       = []
embed_texts = []  # what we actually embed — the prompt, not the code

for i, entry in enumerate(entries):
    doc_id   = f"ex_{i:05d}"
    doc_text = build_document(entry)
    meta     = extract_metadata(entry)
    prompt   = entry.get("prompt", "").strip()

    documents.append(doc_text)
    metadatas.append(meta)
    ids.append(doc_id)
    embed_texts.append(prompt)   # embed prompt text for query-to-prompt cosine match

print(f"      Built {len(documents):,} documents.")

# Design type distribution
from collections import Counter
type_dist = Counter(m["design_type"] for m in metadatas)
print("      Design type distribution:")
for dtype, cnt in type_dist.most_common():
    print(f"        {dtype:16s}: {cnt:5d}")


# ---------------------------------------------------------------------------
# 7. Load embedding model
# ---------------------------------------------------------------------------
print(f"[3/5] Loading embedding model from {MODEL_PATH} ...")
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer(str(MODEL_PATH))
print("      Model loaded.")


# ---------------------------------------------------------------------------
# 8. Compute embeddings in batches
# ---------------------------------------------------------------------------
print(f"[4/5] Embedding {len(embed_texts):,} prompts in batches of {BATCH_SIZE} ...")
try:
    from tqdm import tqdm
    _tqdm = tqdm
except ImportError:
    _tqdm = lambda x, **kw: x  # noqa: E731

all_embeddings = []
for start in _tqdm(range(0, len(embed_texts), BATCH_SIZE),
                   desc="embedding", unit="batch"):
    batch = embed_texts[start : start + BATCH_SIZE]
    vecs = emb_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    all_embeddings.extend(vecs.tolist())

print(f"      Done. {len(all_embeddings):,} embedding vectors computed.")


# ---------------------------------------------------------------------------
# 9. Upsert into ChromaDB
# ---------------------------------------------------------------------------
print(f"[5/5] Writing to ChromaDB at {DB_PATH} ...")
import chromadb

DB_PATH.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(DB_PATH))

if RESET:
    try:
        client.delete_collection(COLLECTION)
        print(f"      Deleted existing collection '{COLLECTION}'.")
    except Exception:
        pass  # Collection did not exist yet

col = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# Upsert in batches (ChromaDB has a soft limit per add() call)
CHROMA_BATCH = 512
added = 0
for start in range(0, len(documents), CHROMA_BATCH):
    end = start + CHROMA_BATCH
    col.add(
        documents  = documents[start:end],
        embeddings = all_embeddings[start:end],
        metadatas  = metadatas[start:end],
        ids        = ids[start:end],
    )
    added += end - start
    print(f"      Upserted {min(added, len(documents)):,} / {len(documents):,}", end="\r")

print(f"\n      Collection '{COLLECTION}' now holds {col.count():,} documents.")

# ---------------------------------------------------------------------------
# 10. Quick sanity check
# ---------------------------------------------------------------------------
print("\n[SANITY] Running test queries ...")
test_queries = [
    ("4-bit synchronous counter with enable and reset",  "counter"),
    ("2-to-1 mux combinational logic",                   "mux"),
    ("D flip-flop with synchronous active-high reset",   "dff"),
    ("FSM traffic light controller Moore machine",       "fsm"),
    ("8-bit ALU with add subtract and logic operations", "alu"),
]
for qtext, expected_type in test_queries:
    q_emb = emb_model.encode([qtext], normalize_embeddings=True).tolist()
    res   = col.query(query_embeddings=q_emb, n_results=1,
                      include=["documents", "metadatas", "distances"])
    top_doc  = res["documents"][0][0][:80].replace("\n", " ")
    top_meta = res["metadatas"][0][0]
    top_dist = res["distances"][0][0]
    got_type = top_meta.get("design_type", "?")
    status   = "OK" if got_type == expected_type else f"MISMATCH (expected {expected_type})"
    print(f"  [{status}] '{qtext[:50]}' → type={got_type} dist={top_dist:.3f}")
    print(f"         doc: {top_doc}")

print("\n[DONE] Ingestion complete.")
print(f"       Database : {DB_PATH.resolve()}")
print(f"       Collection: {COLLECTION}")
print(f"       Documents : {col.count():,}")
print()
print("Next step: restart app2.py — it will auto-detect and load this collection.")
