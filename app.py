"""
Verilog AI Assistant — HDL-Sentinel
====================================
Streamlit-based tutoring tool that generates verified Verilog designs using a
fine-tuned Qwen2.5-Coder-32B (W4A16 SFT) model served via vLLM.

Architecture:
  - Single-pass generation: one LLM call produces design + testbench together
  - iverilog verification: compiles + simulates; on failure, one correction
    turn with the error is sent back to the model, up to 3 attempts
  - RAG: 14k Verilog code corpus (primary), 14 built-in examples (fallback),
    PDF theory corpus (concept questions only)
  - Math bouncer: vector-distance check rejects off-topic queries
  - Firebase: email/password auth + per-user Firestore chat history
  - WSL Ubuntu-22.04 runs iverilog -g2012 (SystemVerilog 2012)
"""

# === Core Python ===
import os
import re
import json
import uuid
import time
import subprocess
import tempfile

# === Third-party ===
import streamlit as st
import chromadb
import firebase_admin
import requests
from firebase_admin import credentials, auth, firestore
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from streamlit_ace import st_ace


# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(layout="wide", page_title="Verilog AI Assistant")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model / Database Paths ---
CHROMA_DB_PATH          = os.path.join(SCRIPT_DIR, "chroma_db_pdfs_only")
CODE_RAG_DB_PATH        = os.path.join(SCRIPT_DIR, "chroma_db_code_examples")
EMBEDDING_MODEL_PATH    = os.path.join(SCRIPT_DIR, "embedding_model")
FIREBASE_CRED_PATH      = os.path.join(SCRIPT_DIR, "firebase_service_account.json")
COLLECTION_NAME         = "verilog_pdf_chunks"
CODE_RAG_COLLECTION     = "verilog_code_examples"
BUILTIN_COLLECTION      = "builtin_verilog_examples"

# --- Secrets: env vars -> streamlit secrets -> local fallback ---
def _load_env(key: str, fallback: str = "") -> str:
    v = os.environ.get(key, "").strip()
    if v:
        return v
    try:
        v = st.secrets.get(key, "")
    except Exception:
        v = ""
    return v or fallback

VLLM_BASE_URL   = _load_env("VLLM_BASE_URL", "http://3.148.67.20/v1")
VLLM_API_KEY    = _load_env("VLLM_API_KEY",
                             "b177bf9dcf72131d2ce9eaf0eba1d23cb3036af13e290dc9719f0ceaa9f382f6")
VERBOSE_DEBUG   = bool(_load_env("VERBOSE_DEBUG", ""))

# LLM BACKEND SELECTION
# ─────────────────────
# If OPENAI_API_KEY is set, use OpenAI API (gpt-4o-mini by default — ~$30/year
# for 30k queries, production-grade reliability).
# Otherwise, fall back to the vLLM server (fine-tuned Qwen2.5-Coder-32B W4A16).
# Override the model with LLM_MODEL_NAME env var (e.g., "gpt-4o", "gpt-4.1").
_OPENAI_KEY     = _load_env("OPENAI_API_KEY", "")
USE_OPENAI_API  = bool(_OPENAI_KEY)
LLM_MODEL_NAME  = _load_env("LLM_MODEL_NAME",
                             "gpt-4o-mini" if USE_OPENAI_API else "/model")

# Firebase Web API Key — needed for password verification (different from
# service account JSON). Get this from Firebase Console → Project Settings →
# Your apps → Web API Key.
FIREBASE_WEB_API_KEY = _load_env("FIREBASE_WEB_API_KEY", "")

# --- Generation Budget ---
VLLM_MAX_CONTEXT        = 8192
MAX_COMPLETION_TOKENS   = 4096
TOKEN_SAFETY_MARGIN     = 100
CHARS_PER_TOKEN         = 2.0
MAX_CONTEXT_CHARS       = 3000       # hard ceiling for RAG context

# --- RAG Parameters ---
N_RESULTS               = 10
MAX_CONTEXT_CHUNKS      = 3
MAX_DISTANCE_THRESHOLD  = 0.4        # bouncer cutoff
CODE_RAG_TOP_K          = 2
CODE_RAG_MAX_DIST       = 0.45

# --- Simulation ---
WSL_DISTRO              = "Ubuntu-22.04"
SIMULATION_TIMEOUT      = 30
MAX_CORRECTION_ATTEMPTS = 3
MAX_ERROR_CHARS         = 800
FAIL_MARKERS            = ["TB_FAIL", "MISMATCH", "ASSERTION FAILED"]

# --- UI Rate Limit ---
COOLDOWN_SECONDS        = 15
MAX_PROMPT_CHARS        = 1000


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================
CODE_SYSTEM_PROMPT = (
    "You are a Verilog RTL design engineer. For every code request, output EXACTLY this structure:\n"
    "1. A brief intro (what the design is, 1-2 sentences).\n"
    "2. Your approach as 2-3 bullet points.\n"
    "3. A heading '#### Design Module' followed by ONE ```verilog code block "
    "containing the complete synthesizable DUT module.\n"
    "4. A heading '#### Testbench Code' followed by ONE ```verilog code block "
    "containing a self-checking testbench that prints 'TB_PASS' on success and "
    "'TB_FAIL: <reason>' on any assertion failure.\n"
    "5. A brief 'How it works' paragraph.\n\n"
    "ABSOLUTE RULES — violations cause compilation failure:\n"
    "- The FIRST ```verilog block is the DESIGN, the SECOND is the TESTBENCH. "
    "Do not swap them.\n"
    "- DUT instantiation syntax: `module_name dut (.port1(sig1), .port2(sig2), .portN(sigN));` "
    "— ONE opening paren after `dut`, ONE closing paren at the end. "
    "NEVER write `.b(b)), .y(y))` — that extra `)` is a syntax error.\n"
    "- Design and testbench port names MUST match exactly (same names, same widths).\n"
    "- Do NOT use undeclared identifiers like WIDTH, CLK_PERIOD, known_pattern — "
    "declare every parameter with `localparam` or use hardcoded literals (e.g. `#10` "
    "instead of `#CLK_PERIOD`).\n"
    "- Testbench MUST include `$dumpfile(\"waveform.vcd\"); $dumpvars(0, <tb_module_name>);` "
    "in the initial block.\n"
    "- End the testbench with `$display(\"TB_PASS\"); $finish;` after all checks pass.\n"
    "- Include `initial #20000 begin $display(\"TB_FAIL: timeout\"); $finish; end` as safety.\n\n"
    "TESTBENCH ASSERTION STYLE — iverilog-compatible only:\n"
    "- Use ONLY `if (signal !== expected) begin $display(\"TB_FAIL: <reason>\"); $finish; end` "
    "for checks. This is the required pattern.\n"
    "- DO NOT use SystemVerilog `assert(...)` statements. iverilog does not support them reliably.\n"
    "- DO NOT use `$fatal(...)`. Use `$display(\"TB_FAIL: ...\"); $finish;` instead.\n"
    "- DO NOT use `$error(...)`, `$warning(...)`, or any other SVA constructs.\n"
    "- DO NOT use `assume`, `cover`, or `property` statements.\n"
    "- Use `reg` for testbench signals (not `logic`) for maximum iverilog compatibility.\n\n"
    "DESIGN SIMPLICITY — do NOT over-engineer:\n"
    "- If asked for a simple combinational circuit (gate, mux, decoder, adder), "
    "use a single `assign` statement. No clock, no register, no always block.\n"
    "- Only add a clock if the user EXPLICITLY asks for 'registered' or 'sequential' or names clocks.\n"
    "- A '2-to-1 mux' is combinational by default: `assign y = sel ? b : a;` — that's the entire design.\n"
    "- An AND gate is `assign y = a & b;`. Don't add always blocks to simple gates.\n\n"
    "Timing rules:\n"
    "- Sequential designs: use `@(posedge clk); #1;` before every output check. "
    "Apply reset this way: `rst=1; @(posedge clk); #1; rst=0; @(posedge clk); #1;`\n"
    "- Combinational designs: use `#10; #1;` to apply inputs and sample outputs. "
    "NO clock, NO reset, NO always block in a combinational testbench."
)

CONCEPT_SYSTEM_PROMPT = (
    "You are a Verilog RTL design engineer answering a conceptual question.\n"
    "Give a clear, concise prose explanation. Include a small code snippet only "
    "if it aids understanding. Do NOT generate full modules or testbenches.\n\n"
    "For off-topic (non-HDL) questions, reply: "
    "'I can only help with Verilog and HDL topics.'"
)

CORRECTION_SYSTEM_PROMPT = (
    "You are fixing a specific Verilog error. Output the corrected design module "
    "and testbench in two separate ```verilog code blocks. "
    "Do not explain. Fix only what the error describes. Keep port names consistent "
    "between design and testbench."
)


# =============================================================================
# INTENT DETECTION
# =============================================================================
CODE_KEYWORDS = [
    'design', 'write', 'implement', 'create', 'build', 'generate', 'code',
    'module', 'testbench', 'make', 'need', 'give me', 'show me',
    'counter', 'mux', 'multiplexer', 'adder', 'alu', 'register', 'shifter',
    'flip-flop', 'flipflop', 'flop', 'latch', 'ram', 'fifo', 'uart',
    'encoder', 'decoder', 'fsm', 'state machine', 'shift register',
    'half adder', 'full adder', 'priority encoder', 'barrel shifter',
    'lfsr', 'johnson',
    'debug', 'fix', 'error',
]

CONCEPT_KEYWORDS = [
    'what is', 'what are', 'explain', 'difference between', 'compare',
    'theory', 'concept', 'how does', 'why does', 'define', 'meaning of',
    'advantages', 'disadvantages', 'when to use', 'purpose of',
    'tell me about', 'summarize', 'summary of',
    'lines about', 'paragraph about', 'briefly describe',
]


def is_code_request(query: str) -> bool:
    q = query.lower()
    if any(kw in q for kw in CONCEPT_KEYWORDS):
        return False
    return any(kw in q for kw in CODE_KEYWORDS)


# =============================================================================
# BUILT-IN CURATED EXAMPLES (fallback when 14k corpus misses)
# =============================================================================
BUILTIN_EXAMPLES = [
    {"id": "and_gate", "query": "2-input AND gate",
     "text": "module and_gate(input a, input b, output y);\n    assign y = a & b;\nendmodule"},
    {"id": "or_gate", "query": "2-input OR gate",
     "text": "module or_gate(input a, input b, output y);\n    assign y = a | b;\nendmodule"},
    {"id": "nand_gate", "query": "2-input NAND gate",
     "text": "module nand_gate(input a, input b, output y);\n    assign y = ~(a & b);\nendmodule"},
    {"id": "nor_gate", "query": "2-input NOR gate",
     "text": "module nor_gate(input a, input b, output y);\n    assign y = ~(a | b);\nendmodule"},
    {"id": "xor_gate", "query": "2-input XOR gate",
     "text": "module xor_gate(input a, input b, output y);\n    assign y = a ^ b;\nendmodule"},
    {"id": "mux2to1", "query": "2-to-1 multiplexer",
     "text": "module mux2to1(input a, input b, input sel, output y);\n    assign y = sel ? b : a;\nendmodule"},
    {"id": "mux4to1", "query": "4-to-1 multiplexer with 2-bit select",
     "text": "module mux4to1(input [3:0] din, input [1:0] sel, output y);\n    assign y = din[sel];\nendmodule"},
    {"id": "half_adder", "query": "half adder",
     "text": "module half_adder(input a, input b, output sum, output cout);\n    assign sum = a ^ b;\n    assign cout = a & b;\nendmodule"},
    {"id": "full_adder", "query": "full adder with carry in and carry out",
     "text": "module full_adder(input a, input b, input cin, output sum, output cout);\n    assign sum = a ^ b ^ cin;\n    assign cout = (a & b) | (cin & (a ^ b));\nendmodule"},
    {"id": "dff_sync", "query": "D flip-flop with synchronous reset",
     "text": "module dff(input clk, input rst, input d, output reg q);\n    always @(posedge clk)\n        if (rst) q <= 1'b0;\n        else q <= d;\nendmodule"},
    {"id": "tff", "query": "T flip-flop with synchronous reset",
     "text": "module tff(input clk, input rst, input t, output reg q);\n    always @(posedge clk)\n        if (rst) q <= 1'b0;\n        else if (t) q <= ~q;\nendmodule"},
    {"id": "jkff", "query": "JK flip-flop with synchronous reset",
     "text": "module jkff(input clk, input rst, input j, input k, output reg q);\n    always @(posedge clk)\n        if (rst) q <= 1'b0;\n        else case ({j,k})\n            2'b01: q <= 1'b0;\n            2'b10: q <= 1'b1;\n            2'b11: q <= ~q;\n        endcase\nendmodule"},
    {"id": "counter_4bit", "query": "4-bit binary up counter with enable and reset",
     "text": "module counter_4bit(input clk, input rst, input en, output reg [3:0] cnt);\n    always @(posedge clk)\n        if (rst) cnt <= 4'd0;\n        else if (en) cnt <= cnt + 1;\nendmodule"},
    {"id": "shift_reg_4bit", "query": "4-bit serial-in parallel-out shift register",
     "text": "module shift_reg(input clk, input rst, input sin, output reg [3:0] q);\n    always @(posedge clk)\n        if (rst) q <= 4'b0;\n        else q <= {q[2:0], sin};\nendmodule"},
    {"id": "priority_encoder_4to2", "query": "4-to-2 priority encoder",
     "text": "module priority_encoder(input [3:0] din, output reg [1:0] y, output reg valid);\n    always @(*) begin\n        valid = |din;\n        casez (din)\n            4'b1???: y = 2'd3;\n            4'b01??: y = 2'd2;\n            4'b001?: y = 2'd1;\n            4'b0001: y = 2'd0;\n            default: y = 2'd0;\n        endcase\n    end\nendmodule"},
    {"id": "decoder_2to4", "query": "2-to-4 decoder with enable",
     "text": "module decoder_2to4(input [1:0] in, input en, output reg [3:0] out);\n    always @(*)\n        if (!en) out = 4'b0;\n        else case (in)\n            2'b00: out = 4'b0001;\n            2'b01: out = 4'b0010;\n            2'b10: out = 4'b0100;\n            2'b11: out = 4'b1000;\n        endcase\nendmodule"},
]


# =============================================================================
# INITIALIZATION (cached)
# =============================================================================
@st.cache_resource
def _download_hf_data():
    """On HF Spaces: download data files from private dataset repo if missing."""
    if not os.environ.get("SPACE_ID"):
        return
    needs = not (
        os.path.isdir(EMBEDDING_MODEL_PATH)
        and os.path.isdir(CHROMA_DB_PATH)
        and os.path.isdir(CODE_RAG_DB_PATH)
    )
    if not needs:
        return
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=os.environ.get("HF_DATASET_REPO", "Akshat1661/hdl-sentinel-data"),
            repo_type="dataset",
            local_dir=SCRIPT_DIR,
            token=os.environ.get("HF_TOKEN") or None,
        )
        print("[HF] Data files downloaded successfully.")
    except Exception as e:
        print(f"[HF] Data download failed: {e}")


@st.cache_resource
def initialize_resources():
    """Load Firebase, embedding model, ChromaDBs, LLM client. Cached — runs once."""
    _download_hf_data()
    resources = {
        "firebase_db": None,
        "embedding_model": None,
        "theory_collection": None,
        "code_rag_collection": None,
        "builtin_collection": None,
        "llm_client": None,
    }

    # Firebase — env var JSON wins, then file path
    fb_env = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    fb_cred = None
    if fb_env:
        try:
            fb_cred = credentials.Certificate(json.loads(fb_env))
        except Exception as e:
            st.error(f"Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
    elif os.path.exists(FIREBASE_CRED_PATH):
        try:
            fb_cred = credentials.Certificate(FIREBASE_CRED_PATH)
        except Exception as e:
            st.error(f"Firebase credential file invalid: {e}")
    if fb_cred is not None:
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app(fb_cred)
            resources["firebase_db"] = firestore.client()
        except Exception as e:
            st.error(f"Firebase initialization failed: {e}")

    # Embedding model: prefer local folder if present, else download from HF Hub.
    # The HF Hub version is the same all-MiniLM-L6-v2 model, free, cached on HF Spaces.
    _HF_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_HF_ID",
                                          "sentence-transformers/all-MiniLM-L6-v2")
    if os.path.isdir(EMBEDDING_MODEL_PATH):
        try:
            resources["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_PATH)
            print(f"[RAG] Embedding model loaded from local: {EMBEDDING_MODEL_PATH}")
        except Exception as e:
            st.error(f"Failed to load local embedding model: {e}")
    else:
        try:
            resources["embedding_model"] = SentenceTransformer(_HF_EMBEDDING_MODEL)
            print(f"[RAG] Embedding model loaded from HF Hub: {_HF_EMBEDDING_MODEL}")
        except Exception as e:
            st.error(f"Failed to load embedding model from HF Hub: {e}")

    # Theory ChromaDB (PDF corpus)
    if os.path.isdir(CHROMA_DB_PATH):
        try:
            cli = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            resources["theory_collection"] = cli.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            print(f"[RAG] Theory corpus unavailable: {e}")

    # 14k code RAG
    if os.path.isdir(CODE_RAG_DB_PATH):
        try:
            cli = chromadb.PersistentClient(path=CODE_RAG_DB_PATH)
            resources["code_rag_collection"] = cli.get_collection(name=CODE_RAG_COLLECTION)
            print(f"[RAG] Code corpus: {resources['code_rag_collection'].count():,} documents")
        except Exception as e:
            print(f"[RAG] Code corpus unavailable: {e}")

    # Built-in examples — in-memory via ChromaDB (refreshed each startup)
    emb = resources["embedding_model"]
    if emb is not None:
        try:
            mem_cli = chromadb.EphemeralClient()
            col = mem_cli.create_collection(
                name=BUILTIN_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
            embs = emb.encode([ex["query"] for ex in BUILTIN_EXAMPLES],
                              normalize_embeddings=True).tolist()
            col.add(
                documents=[ex["text"] for ex in BUILTIN_EXAMPLES],
                embeddings=embs,
                ids=[ex["id"] for ex in BUILTIN_EXAMPLES],
            )
            resources["builtin_collection"] = col
            print(f"[RAG] Built-in examples: {len(BUILTIN_EXAMPLES)} seeded")
        except Exception as e:
            print(f"[RAG] Built-in examples unavailable: {e}")

    # vLLM client OR OpenAI API client (based on env var OPENAI_API_KEY)
    if USE_OPENAI_API:
        resources["llm_client"] = OpenAI(api_key=_OPENAI_KEY)
        print(f"[LLM] Using OpenAI API with model: {LLM_MODEL_NAME}")
    else:
        resources["llm_client"] = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        print(f"[LLM] Using vLLM server at {VLLM_BASE_URL} with model: {LLM_MODEL_NAME}")

    return resources


# =============================================================================
# FIREBASE HELPERS
# =============================================================================
def firebase_signin_password(email: str, password: str) -> dict:
    """Verify email+password via Firebase Auth REST API.

    Returns {"ok": True, "uid": ..., "email": ...} on success,
    or   {"ok": False, "error": "<message>"} on failure.

    Requires FIREBASE_WEB_API_KEY to be configured.
    """
    if not FIREBASE_WEB_API_KEY:
        return {"ok": False, "error": "FIREBASE_WEB_API_KEY not configured"}
    url = ("https://identitytoolkit.googleapis.com/v1/accounts:"
           f"signInWithPassword?key={FIREBASE_WEB_API_KEY}")
    try:
        r = requests.post(url, json={"email": email, "password": password,
                                     "returnSecureToken": True}, timeout=10)
        data = r.json()
        if r.status_code == 200 and "localId" in data:
            return {"ok": True, "uid": data["localId"], "email": data.get("email", email)}
        # Map common error codes to friendly messages
        err_msg = data.get("error", {}).get("message", "Login failed")
        friendly = {
            "EMAIL_NOT_FOUND":   "No account found with that email.",
            "INVALID_PASSWORD":  "Incorrect password.",
            "INVALID_LOGIN_CREDENTIALS": "Invalid email or password.",
            "USER_DISABLED":     "This account has been disabled.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts — please try again later.",
        }.get(err_msg, err_msg)
        return {"ok": False, "error": friendly}
    except requests.Timeout:
        return {"ok": False, "error": "Firebase auth request timed out"}
    except Exception as e:
        return {"ok": False, "error": f"Firebase auth error: {e}"}


def firebase_signup_password(email: str, password: str) -> dict:
    """Create a new user via Firebase Auth REST API.

    Returns {"ok": True, "uid": ..., "email": ...} on success,
    or   {"ok": False, "error": "<message>"} on failure.
    """
    if not FIREBASE_WEB_API_KEY:
        return {"ok": False, "error": "FIREBASE_WEB_API_KEY not configured"}
    url = ("https://identitytoolkit.googleapis.com/v1/accounts:"
           f"signUp?key={FIREBASE_WEB_API_KEY}")
    try:
        r = requests.post(url, json={"email": email, "password": password,
                                     "returnSecureToken": True}, timeout=10)
        data = r.json()
        if r.status_code == 200 and "localId" in data:
            return {"ok": True, "uid": data["localId"], "email": data.get("email", email)}
        err_msg = data.get("error", {}).get("message", "Signup failed")
        friendly = {
            "EMAIL_EXISTS":      "An account with this email already exists. Try logging in.",
            "OPERATION_NOT_ALLOWED": "Email/password signup is disabled in Firebase settings.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts — please try again later.",
            "WEAK_PASSWORD : Password should be at least 6 characters":
                "Password must be at least 6 characters.",
        }.get(err_msg, err_msg)
        return {"ok": False, "error": friendly}
    except requests.Timeout:
        return {"ok": False, "error": "Firebase signup request timed out"}
    except Exception as e:
        return {"ok": False, "error": f"Firebase signup error: {e}"}


def get_chat_sessions(db, uid):
    if not db or not uid:
        return []
    try:
        ref = (db.collection('users').document(uid).collection('chats')
               .order_by("last_updated", direction=firestore.Query.DESCENDING).stream())
        return [{"id": doc.id, "title": doc.to_dict().get("title", "Untitled")} for doc in ref]
    except Exception:
        return []


def load_messages(db, uid, chat_id):
    if not db or not uid or not chat_id:
        return []
    try:
        doc = db.collection('users').document(uid).collection('chats').document(chat_id).get()
        return doc.to_dict().get("messages", []) if doc.exists else []
    except Exception:
        return []


def save_messages(db, uid, chat_id, messages):
    if not db or not uid or not chat_id or not messages:
        return
    try:
        ref = db.collection('users').document(uid).collection('chats').document(chat_id)
        existing = ref.get()
        if existing.exists and existing.to_dict().get("title", "Untitled") != "Untitled":
            title = existing.to_dict().get("title")
        else:
            title = next((m['content'][:50] for m in messages if m['role'] == 'user'), "Untitled")
        ref.set({"title": title, "messages": messages,
                 "last_updated": firestore.SERVER_TIMESTAMP}, merge=True)
    except Exception as e:
        st.error(f"Failed to save chat: {e}")


def delete_chat(db, uid, chat_id):
    if not db or not uid or not chat_id:
        return
    try:
        db.collection('users').document(uid).collection('chats').document(chat_id).delete()
    except Exception as e:
        st.error(f"Failed to delete chat: {e}")


# =============================================================================
# RAG RETRIEVAL
# =============================================================================
def get_embedding(text, model):
    if model is None or not text:
        return None
    try:
        return model.encode([text], normalize_embeddings=True)
    except Exception:
        return None


def retrieve_theory(col, query_emb, k=MAX_CONTEXT_CHUNKS):
    """Returns (docs, closest_dist) from theory corpus. Closest dist is for bouncer."""
    if col is None or query_emb is None:
        return [], 999.0
    try:
        res = col.query(query_embeddings=query_emb.tolist(), n_results=N_RESULTS,
                        include=["documents", "distances"])
        docs  = res.get("documents", [[]])[0]
        dists = res.get("distances",  [[]])[0]
        closest = min(dists) if dists else 999.0
        return docs[:k], closest
    except Exception:
        return [], 999.0


def retrieve_code_examples(code_col, builtin_col, emb_model, query, k=CODE_RAG_TOP_K):
    """Try 14k corpus first, fall back to built-in curated examples."""
    if emb_model is None:
        return []
    try:
        q_emb = emb_model.encode([query], normalize_embeddings=True).tolist()
    except Exception:
        return []

    # 14k corpus
    if code_col is not None:
        try:
            res = code_col.query(query_embeddings=q_emb, n_results=k,
                                 include=["documents", "distances"])
            docs  = res.get("documents", [[]])[0]
            dists = res.get("distances",  [[]])[0]
            hits = [d for d, dist in zip(docs, dists) if dist < CODE_RAG_MAX_DIST]
            if hits:
                if VERBOSE_DEBUG:
                    print(f"[Code RAG] 14k corpus: {len(hits)} hit(s)")
                return hits
        except Exception as e:
            if VERBOSE_DEBUG:
                print(f"[Code RAG] 14k query failed: {e}")

    # Built-in fallback
    if builtin_col is not None:
        try:
            res = builtin_col.query(query_embeddings=q_emb, n_results=k,
                                    include=["documents", "distances"])
            docs  = res.get("documents", [[]])[0]
            dists = res.get("distances",  [[]])[0]
            hits = [d for d, dist in zip(docs, dists) if dist < 0.6]  # looser for builtin
            if hits:
                if VERBOSE_DEBUG:
                    print(f"[Code RAG] built-in: {len(hits)} hit(s)")
                return hits
        except Exception:
            pass

    return []


# =============================================================================
# SIMULATION
# =============================================================================
# Platform auto-detection:
#   - Windows: call iverilog via WSL (sys.platform == 'win32')
#   - Linux/Mac (including HF Spaces): call iverilog directly
import sys
USE_WSL = sys.platform == "win32"


def win_to_wsl_path(windows_path: str) -> str:
    """C:\\foo\\bar -> /mnt/c/foo/bar (Windows + WSL only)"""
    p = os.path.abspath(windows_path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return p


def _iverilog_cmd(*args) -> list:
    """Build iverilog command — WSL wrapper on Windows, direct on Linux."""
    if USE_WSL:
        return ["wsl", "-d", WSL_DISTRO, "iverilog"] + list(args)
    return ["iverilog"] + list(args)


def _vvp_cmd(tmp_dir: str, vvp_file: str) -> list:
    """Build vvp command — WSL wrapper on Windows, direct on Linux."""
    if USE_WSL:
        return ["wsl", "-d", WSL_DISTRO, "sh", "-c",
                f"cd '{tmp_dir}' && vvp '{vvp_file}'"]
    return ["sh", "-c", f"cd '{tmp_dir}' && vvp '{vvp_file}'"]


def run_simulation(design_code: str, tb_code: str) -> dict:
    """Compile + simulate via iverilog. Returns result dict.
    Auto-uses WSL on Windows, direct iverilog on Linux (HF Spaces / cloud).
    """
    result = {
        "success": False, "compile_error": False, "functional_fail": False,
        "env_error": False, "log": "", "simulation_output": "",
        "raw_errors": "", "vcd_data": None,
    }

    if not design_code.strip() or not tb_code.strip():
        result["raw_errors"] = "Missing design or testbench code."
        return result

    with tempfile.TemporaryDirectory() as tmp:
        design_path = os.path.join(tmp, "design.v")
        tb_path     = os.path.join(tmp, "tb_design.v")
        vvp_path    = os.path.join(tmp, "simulation.vvp")
        vcd_path    = os.path.join(tmp, "waveform.vcd")

        with open(design_path, "w", encoding="utf-8") as f:
            f.write(design_code)
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(tb_code)

        # Translate paths for the shell environment
        if USE_WSL:
            design_arg = win_to_wsl_path(design_path)
            tb_arg     = win_to_wsl_path(tb_path)
            vvp_arg    = win_to_wsl_path(vvp_path)
            tmp_arg    = win_to_wsl_path(tmp)
        else:
            design_arg = design_path
            tb_arg     = tb_path
            vvp_arg    = vvp_path
            tmp_arg    = tmp

        # Compile
        try:
            proc = subprocess.run(
                _iverilog_cmd("-g2012", "-o", vvp_arg, design_arg, tb_arg),
                capture_output=True, text=True, timeout=SIMULATION_TIMEOUT,
                check=False, encoding="utf-8",
            )
            result["log"] = f"--- Compile ---\n{proc.stdout}\n{proc.stderr}\n"
            if proc.returncode != 0:
                result["compile_error"] = True
                result["raw_errors"] = (proc.stderr or proc.stdout)[:MAX_ERROR_CHARS]
                return result
        except FileNotFoundError:
            result["env_error"] = True
            result["raw_errors"] = (
                f"iverilog not found (via {'WSL/'+WSL_DISTRO if USE_WSL else 'system PATH'}). "
                "Install Icarus Verilog."
            )
            return result
        except subprocess.TimeoutExpired:
            result["env_error"] = True
            result["raw_errors"] = f"Compilation timeout ({SIMULATION_TIMEOUT}s)."
            return result
        except Exception as e:
            result["compile_error"] = True
            result["raw_errors"] = f"Compile failure: {e}"
            return result

        # Simulate
        try:
            proc = subprocess.run(
                _vvp_cmd(tmp_arg, vvp_arg),
                capture_output=True, text=True, timeout=SIMULATION_TIMEOUT,
                check=False, encoding="utf-8",
            )
            result["log"] += f"\n--- Simulate ---\n{proc.stdout}\n{proc.stderr}\n"
            result["simulation_output"] = proc.stdout

            sim_upper = proc.stdout.upper()
            fails = [l for l in proc.stdout.splitlines()
                     if any(m.lower() in l.lower() for m in FAIL_MARKERS)]
            if fails:
                result["functional_fail"] = True
                result["raw_errors"] = "FUNCTIONAL FAIL:\n" + "\n".join(fails[:5])
                result["raw_errors"] = result["raw_errors"][:MAX_ERROR_CHARS]
            elif "TB_PASS" in sim_upper:
                result["success"] = True
            elif proc.returncode == 0 and sim_upper.strip():
                # Simulation ran to completion without markers — treat as pass
                result["success"] = True
            else:
                result["functional_fail"] = True
                tail = "\n".join(proc.stdout.splitlines()[-10:])
                result["raw_errors"] = f"Inconclusive:\n{tail}"[:MAX_ERROR_CHARS]

            # Grab any VCD
            vcd_files = [f for f in os.listdir(tmp) if f.endswith(".vcd")]
            if vcd_files:
                chosen = vcd_path if os.path.exists(vcd_path) else os.path.join(tmp, vcd_files[0])
                with open(chosen, "rb") as f:
                    result["vcd_data"] = f.read()
        except subprocess.TimeoutExpired:
            result["env_error"] = True
            result["raw_errors"] = f"Simulation timeout ({SIMULATION_TIMEOUT}s)."
        except Exception as e:
            result["functional_fail"] = True
            result["raw_errors"] = f"Simulation failure: {e}"

    return result


# =============================================================================
# CODE EXTRACTION & POSTPROCESS
# =============================================================================
def looks_like_tb(code: str) -> bool:
    """Heuristic: is this code a testbench (not a design)?"""
    if not code:
        return False
    has_initial_display = bool(re.search(r'\binitial\b', code)) and \
                          bool(re.search(r'\$display|\$finish|\$dumpfile', code))
    has_dut_instance    = bool(re.search(r'\b(?:dut|uut)\s*\(', code, re.IGNORECASE))
    has_tb_module_name  = bool(re.search(r'\bmodule\s+(?:tb_|\w*_?tb\b)', code, re.IGNORECASE))
    # Strong signal: TB-like module name OR (initial+display AND dut instance)
    return has_tb_module_name or (has_initial_display and has_dut_instance)


def fix_common_verilog_bugs(code: str) -> str:
    """Auto-repair systematic model output bugs observed in the 35-prompt test.

    Idempotent — safe to call multiple times.

    Bugs fixed:
      1. Extra `)` in DUT instantiation: "dut (.a(a), .b(b)), .y(y))" →
         "dut (.a(a), .b(b), .y(y))"
      2. "$0finish" typo → "$finish"
      3. Trailing ")" at end of case/if statements: "}))" → "})"
    """
    if not code:
        return code
    # Extra ")" between port connections:  "),   .port(" should be "),   .port("
    # but sometimes model emits:  ")),  .port("  →  "),  .port("
    code = re.sub(r'\)\)(\s*,\s*\.[A-Za-z_]\w*\s*\()', r')\1', code)
    # Trailing double paren at end of DUT instantiation ";)));"  → "));"
    code = re.sub(r'\)\)\);', r'));', code)
    # "$0finish" typo (seen in up-down counter)
    code = re.sub(r'\$0finish\b', '$finish', code)
    # Stray ")" before "begin" after condition: "if (x)) begin" → "if (x) begin"
    code = re.sub(r'(if\s*\([^()]*\))\)', r'\1', code)
    code = re.sub(r'(case\s*\([^()]*\))\)', r'\1', code)
    return code


def has_excessive_duplication(code: str) -> bool:
    """Detect runaway duplicated blocks (seen in Johnson counter — 40+ repeats).

    Returns True if ANY substantial initial-block fragment appears 5+ times.
    """
    if not code or len(code) < 1000:
        return False
    # Count distinct "initial begin" occurrences — a TB rarely has more than 3-4
    initials = re.findall(r'\binitial\b\s+begin\b', code)
    if len(initials) >= 8:
        return True
    # Check for repeated long lines (model degeneracy signature)
    lines = [l.strip() for l in code.splitlines() if len(l.strip()) > 30]
    from collections import Counter
    if lines:
        most_common, cnt = Counter(lines).most_common(1)[0]
        if cnt >= 6:  # same 30+ char line repeated 6+ times = degeneracy
            return True
    return False


def has_undefined_identifiers(code: str) -> bool:
    """Detect obviously-undefined identifiers the model hallucinates.

    Returns True if code uses common fake placeholder names without declaring them.
    These were seen in modulo-10, modulo-6, JK FF, BCD, sequence detector.
    """
    if not code:
        return False
    # Skip if identifiers are declared via localparam/parameter/reg/integer/wire
    decl_keywords = ('localparam', 'parameter', 'integer', '`define')
    fake_names = ('WIDTH', 'CLK_PERIOD', 'known_pattern', 'known_pattern_width',
                  'known_pattern_expected_final_value')
    for name in fake_names:
        # Check if name is USED
        if re.search(r'\b' + re.escape(name) + r'\b', code):
            # Check if name is DECLARED anywhere
            declared = False
            for kw in decl_keywords:
                if re.search(kw + r'\s+(?:\[[^\]]+\]\s*)?' + re.escape(name) + r'\b', code):
                    declared = True
                    break
            if not declared:
                return True
    return False


def extract_code_blocks(text: str):
    """Extract design + testbench code blocks. Returns (design, testbench).

    Strategy:
      1. Find all ```verilog blocks with module+endmodule
      2. Classify each: design-like vs TB-like (looks_like_tb)
      3. Assign slots: design → first non-TB, testbench → first TB
      4. If model swapped them under mis-labeled headings, the classification
         still puts them in the right slot.
      5. Apply auto-fix (fix_common_verilog_bugs) to both before returning.
    """
    if not text:
        return "", ""

    # Collect all complete code blocks from all fence types
    all_blocks = []

    # ```verilog fenced
    for b in re.findall(r'```verilog\s*\n(.*?)```', text, re.DOTALL):
        bs = b.strip()
        if 'module' in bs and 'endmodule' in bs:
            all_blocks.append(bs)

    # Plain ``` fenced — only if no verilog-fenced blocks found
    if not all_blocks:
        for b in re.findall(r'```[^\n]*\n(.*?)```', text, re.DOTALL):
            bs = b.strip()
            if 'module' in bs and 'endmodule' in bs:
                all_blocks.append(bs)

    # Bare module/endmodule scan — only if still nothing
    if not all_blocks:
        for b in re.findall(r'((?:`timescale[^\n]*\n\s*)?module\b[\s\S]*?endmodule)', text):
            bs = b.strip()
            if bs not in all_blocks:
                all_blocks.append(bs)

    # Filter out excessively duplicated (runaway) blocks early
    all_blocks = [b for b in all_blocks if not has_excessive_duplication(b)]

    if not all_blocks:
        return "", ""

    # Classify each block as design or testbench
    design_candidates = [b for b in all_blocks if not looks_like_tb(b)]
    tb_candidates     = [b for b in all_blocks if looks_like_tb(b)]

    design = design_candidates[0] if design_candidates else ""
    tb     = tb_candidates[0]     if tb_candidates     else ""

    # Fallback: if we only got TBs (no design), keep first block as-is for design
    if not design and len(all_blocks) >= 2:
        design = all_blocks[0]
        tb     = all_blocks[1]
    elif not design and all_blocks:
        design = all_blocks[0]

    # If TB still empty but we have 2+ blocks, use second
    if not tb and len(all_blocks) >= 2:
        tb = all_blocks[1] if all_blocks[1] != design else (
            all_blocks[0] if all_blocks[0] != design else "")

    # Apply automatic bug fixes
    design = fix_common_verilog_bugs(design)
    tb     = fix_common_verilog_bugs(tb)

    return design, tb


def ensure_vcd_dump(tb_code: str) -> str:
    """Inject $dumpfile/$dumpvars if missing — Python-side guarantee for waveform."""
    if '$dumpfile' in tb_code:
        return tb_code
    m = re.search(r'\bmodule\s+(\w+)', tb_code)
    mod_name = m.group(1) if m else "tb"
    injected = re.sub(
        r'(\binitial\b\s+begin\b)',
        f'\\1\n        $dumpfile("waveform.vcd");\n        $dumpvars(0, {mod_name});',
        tb_code, count=1
    )
    return injected


def is_complete_module(code: str) -> bool:
    """Sanity check: real module with header + endmodule + non-trivial body.
    Also rejects code with undefined placeholder identifiers (WIDTH, CLK_PERIOD, etc.)
    that the model hallucinates without declaring.
    """
    if not code or not code.strip():
        return False
    header = re.search(r'\bmodule\s+\w+\s*[\(;]', code)
    if not header:
        return False
    if 'endmodule' not in code:
        return False
    end_idx = code.rfind('endmodule')
    if header.start() >= end_idx:
        return False
    body = code[header.end():end_idx]
    if sum(1 for ch in body if not ch.isspace()) < 20:
        return False
    # Reject if it uses common undefined placeholder identifiers
    if has_undefined_identifiers(code):
        return False
    return True


# =============================================================================
# LLM API CALL
# =============================================================================
def call_llm(client, messages, max_tokens):
    """Single LLM call with rate-limit backoff. Returns string or None."""
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=LLM_MODEL_NAME, messages=messages,
                temperature=0.1, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ("401", "403", "unauthorized", "authentication", "forbidden")):
                print(f"[API] Auth error: {e}")
                return "__AUTH_ERROR__"
            if ("429" in msg or "rate" in msg or "quota" in msg) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            print(f"[API] Error: {e}")
            return None
    return None


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


# =============================================================================
# GENERATION PIPELINE
# =============================================================================
def generate_code_response(llm_client, query, code_examples, theory_docs, container):
    """Single-pass code generation with iverilog correction loop.

    Returns the formatted assistant response as markdown string.
    """
    # Build RAG context block — code examples first, theory second
    context_parts = []
    if code_examples:
        code_str = "\n\n---\n\n".join(code_examples)
        if len(code_str) > int(MAX_CONTEXT_CHARS * 0.7):
            code_str = code_str[:int(MAX_CONTEXT_CHARS * 0.7)]
        context_parts.append(
            "[Reference code examples — match this style and port conventions]\n"
            + code_str
        )
    if theory_docs:
        theory_str = "\n\n".join(theory_docs[:1])
        remaining = MAX_CONTEXT_CHARS - sum(len(p) for p in context_parts)
        if remaining > 200:
            context_parts.append(
                "[Technical reference]\n" + theory_str[:remaining]
            )
    context = ("\n\n".join(context_parts) + "\n\n") if context_parts else ""

    user_content = f"{context}Request: {query}"

    messages = [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    max_out = max(min(MAX_COMPLETION_TOKENS,
                      VLLM_MAX_CONTEXT - input_tokens - TOKEN_SAFETY_MARGIN), 500)

    container.info("Generating Verilog design...")
    full_response = call_llm(llm_client, messages, max_out)

    if full_response == "__AUTH_ERROR__":
        return "**API authentication error.** Check that `VLLM_API_KEY` is set correctly."
    if full_response is None:
        return "**Model communication error.** Please try again in a moment."

    # Extract + verify
    design_code, tb_code = extract_code_blocks(full_response)

    if not is_complete_module(design_code):
        return (
            "I wasn't able to produce a complete Verilog module for that request. "
            "Could you try rephrasing with more detail? For example, instead of "
            "*'make a counter'* try *'design a 4-bit up-counter with enable and "
            "synchronous reset'* — include width, reset style, and control signals."
        )

    if not is_complete_module(tb_code):
        # Design is OK but TB is missing — return design only with a note
        return (
            full_response + "\n\n"
            "_Note: the testbench couldn't be extracted cleanly. You can still "
            "click **Open in Simulator** to edit the design and write your own TB._"
        )

    # Inject VCD dump if model forgot
    tb_code = ensure_vcd_dump(tb_code)

    # === Correction loop: up to 3 attempts ===
    for attempt in range(MAX_CORRECTION_ATTEMPTS + 1):
        if attempt > 0:
            container.info(f"Refining the design (attempt {attempt})...")
        else:
            container.info("Verifying with iverilog...")

        sim = run_simulation(design_code, tb_code)

        if sim["env_error"]:
            container.warning("Simulator unavailable — returning unverified.")
            return _format_response(full_response, design_code, tb_code,
                                    verified=False, note=sim["raw_errors"])

        if sim["success"]:
            container.success("Design verified — all tests passed.")
            return _format_response(full_response, design_code, tb_code, verified=True)

        # Failure — prepare correction turn
        if attempt >= MAX_CORRECTION_ATTEMPTS:
            break

        err_text = sim["raw_errors"]
        err_kind = "compile error" if sim["compile_error"] else "simulation failure"
        correction = (
            f"Your previous design had a {err_kind}:\n"
            f"```\n{err_text}\n```\n"
            f"Output the corrected design module AND testbench in two separate "
            f"```verilog code blocks. Keep port names consistent between them. "
            f"Fix only what the error describes; do not change unrelated parts."
        )

        messages = [
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": full_response},
            {"role": "user", "content": correction},
        ]
        input_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        max_out = max(min(MAX_COMPLETION_TOKENS,
                          VLLM_MAX_CONTEXT - input_tokens - TOKEN_SAFETY_MARGIN), 500)

        new_response = call_llm(llm_client, messages, max_out)
        if new_response == "__AUTH_ERROR__" or new_response is None:
            break

        new_design, new_tb = extract_code_blocks(new_response)
        if is_complete_module(new_design) and is_complete_module(new_tb):
            design_code  = new_design
            tb_code      = ensure_vcd_dump(new_tb)
            full_response = new_response
        # else: retry with same code; might have gotten cut off

    # All attempts exhausted
    container.warning("Couldn't fully verify — showing the best attempt.")
    return _format_response(full_response, design_code, tb_code,
                            verified=False,
                            note="This didn't pass every automated check. "
                                 "Click **Open in Simulator** to inspect it yourself.")


def _format_response(original_response, design_code, tb_code, verified=True, note=""):
    """Preserve the model's prose + swap in the final (possibly corrected) code."""
    # Extract prose before the first code block
    prose_match = re.match(r'^(.*?)(?=#{1,4}\s*Design Module|```verilog)',
                           original_response, re.DOTALL)
    prose = prose_match.group(1).strip() if prose_match else ""
    # Extract "how it works" section if present
    howit_match = re.search(r'(#{1,4}\s*How it works.*?)(?=#{1,4}|\Z)',
                            original_response, re.DOTALL | re.IGNORECASE)
    if not howit_match:
        howit_match = re.search(r'(How it works.*?)(?=#{1,4}|\Z)',
                                original_response, re.DOTALL | re.IGNORECASE)
    howit = howit_match.group(1).strip() if howit_match else ""

    parts = []
    if prose:
        parts.append(prose)
    parts.append(f"#### Design Module{' (best attempt — please review)' if not verified else ''}")
    parts.append(f"```verilog\n{design_code}\n```")
    parts.append("#### Testbench Code")
    parts.append(f"```verilog\n{tb_code}\n```")
    if howit:
        parts.append(howit)
    if verified:
        parts.append("_Tip: click **Open in Simulator** to run it yourself and view the waveform._")
    elif note:
        parts.append(f"_{note}_")

    return "\n\n".join(parts)


def generate_concept_response(llm_client, query, theory_docs, container):
    """Theory/concept question — single call, no code verification."""
    context = ""
    if theory_docs:
        context = "[Reference]\n" + "\n\n".join(theory_docs[:2])[:MAX_CONTEXT_CHARS] + "\n\n"

    messages = [
        {"role": "system", "content": CONCEPT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}{query}"},
    ]
    input_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    max_out = max(min(MAX_COMPLETION_TOKENS,
                      VLLM_MAX_CONTEXT - input_tokens - TOKEN_SAFETY_MARGIN), 300)

    container.info("Thinking...")
    response = call_llm(llm_client, messages, max_out)
    if response == "__AUTH_ERROR__":
        return "**API authentication error.**"
    if response is None:
        return "**Model communication error.** Please try again."
    return response


# =============================================================================
# UI: MESSAGE RENDERING
# =============================================================================
def render_message(msg):
    """Render a chat message with code blocks extracted and placed inline."""
    with st.chat_message(msg["role"]):
        content = msg["content"]
        blocks = re.findall(r'```verilog\s*\n(.*?)```', content, re.DOTALL)
        if not blocks:
            st.markdown(content)
            return

        # Prose before first block
        prose = re.sub(r'```verilog\s*\n.*?```', '', content, flags=re.DOTALL).strip()
        # Also strip "Design Module" / "Testbench Code" heading lines that are now bare
        prose = re.sub(r'^#{1,4}\s*Design Module.*$', '', prose, flags=re.MULTILINE)
        prose = re.sub(r'^#{1,4}\s*Testbench Code.*$', '', prose, flags=re.MULTILINE)
        # Clean up extra blank lines
        prose = re.sub(r'\n{3,}', '\n\n', prose).strip()

        # Find where design vs. other prose falls — split at "How it works"
        howit_split = re.split(r'(?:#{1,4}\s*)?How it works',
                               prose, maxsplit=1, flags=re.IGNORECASE)
        intro = howit_split[0].strip()
        howit = ("How it works" + howit_split[1]).strip() if len(howit_split) > 1 else ""

        if intro:
            st.markdown(intro)

        st.markdown("#### Design Module")
        st.code(blocks[0].strip(), language="verilog")

        if len(blocks) >= 2:
            with st.expander("Testbench Code"):
                st.code(blocks[1].strip(), language="verilog")

        if howit:
            st.markdown(howit)

        # Simulator handoff
        if msg["role"] == "assistant" and len(blocks) >= 2:
            design_snap = blocks[0].strip()
            tb_snap     = blocks[1].strip()

            def _send_to_sim(d=design_snap, t=tb_snap):
                st.session_state.verilog_code    = d
                st.session_state.testbench_code  = t
                st.session_state.ace_key_counter += 1
                st.session_state.nav_radio        = "Simulator Sandbox"

            btn_key = f"sim_{abs(hash(content)) % 10_000_000}"
            st.button("Open in Simulator", key=btn_key, type="secondary",
                      on_click=_send_to_sim)


# =============================================================================
# UI: WAVEFORM (preserved from original — works well)
# =============================================================================
MAX_WAVEFORM_SIGNALS = 24


def parse_vcd(vcd_bytes):
    try:
        text = vcd_bytes.decode("utf-8", errors="replace")
    except Exception:
        return {}, {}
    signals, waveforms = {}, {}
    current_time = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("$var"):
            tok = line.split()
            if len(tok) >= 5:
                try:
                    width = int(tok[2]); vid = tok[3]; name = tok[4].split("[")[0]
                    if vid not in signals:
                        signals[vid] = {"name": name, "width": width}
                        if name not in waveforms:
                            waveforms[name] = []
                except (ValueError, IndexError):
                    pass
        elif line.startswith("#"):
            try:
                current_time = int(line[1:].split()[0])
            except (ValueError, IndexError):
                pass
        elif line[0] in "bB":
            parts = line.split(None, 1)
            if len(parts) == 2:
                val_str = parts[0][1:].lower().replace("x", "0").replace("z", "0")
                vid = parts[1].strip()
                if vid in signals:
                    try:
                        val = int(val_str, 2) if val_str else 0
                    except ValueError:
                        val = 0
                    waveforms[signals[vid]["name"]].append((current_time, val))
        elif len(line) >= 2 and line[0] in "01xXzZ":
            vc, vid = line[0], line[1:].strip()
            if vid in signals:
                waveforms[signals[vid]["name"]].append((current_time, 1 if vc == "1" else 0))
    return signals, {k: v for k, v in waveforms.items() if v}


def render_waveform_chart(vcd_bytes):
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("plotly is required. Run: pip install plotly")
        return
    signals, waveforms = parse_vcd(vcd_bytes)
    if not waveforms:
        st.warning("No VCD data found.")
        return
    names = list(waveforms.keys())
    if len(names) > MAX_WAVEFORM_SIGNALS:
        st.caption(f"Showing first {MAX_WAVEFORM_SIGNALS} of {len(names)} signals.")
        names = names[:MAX_WAVEFORM_SIGNALS]
    t_max = max(t for n in names for t, _ in waveforms[n]) or 1
    LANE_H, LANE_GAP = 1.0, 0.6
    LANE_TOTAL = LANE_H + LANE_GAP
    fig = go.Figure()
    ticks, labels = [], []
    COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
              "#79c0ff", "#56d364", "#ff7b72", "#cae8ff", "#b3d8ff"]
    for i, name in enumerate(names):
        data = waveforms[name]
        sig = next((v for v in signals.values() if v["name"] == name), {"width": 1})
        is_bus = sig["width"] > 1
        y_base = i * LANE_TOTAL
        y_top, y_mid = y_base + LANE_H, y_base + LANE_H / 2
        ticks.append(y_mid); labels.append(name)
        times  = [t for t, _ in data]
        values = [v for _, v in data]
        if times[-1] < t_max:
            times.append(t_max); values.append(values[-1])
        color = COLORS[i % len(COLORS)]
        if not is_bus:
            xs, ys = [], []
            for j in range(len(times) - 1):
                yv = y_top if values[j] else y_base
                xs += [times[j], times[j + 1]]
                ys += [yv, yv]
            x_full, y_full = [], []
            prev = None
            for x, y in zip(xs, ys):
                if prev is not None and y != prev:
                    x_full += [x, x]; y_full += [prev, y]
                x_full.append(x); y_full.append(y); prev = y
            fig.add_trace(go.Scatter(x=x_full, y=y_full, mode="lines",
                line=dict(color=color, width=1.5), name=name, showlegend=False))
        else:
            max_val = max(values) or 1
            delta = max(t_max / 400, 1)
            xs, ys = [], []
            for j in range(len(times) - 1):
                v_norm = (values[j] / max_val) * LANE_H + y_base
                nxt_v = values[min(j + 1, len(values) - 1)]
                nxt_norm = (nxt_v / max_val) * LANE_H + y_base
                xs += [times[j], times[j + 1] - delta, times[j + 1] - delta, times[j + 1]]
                ys += [v_norm, v_norm, nxt_norm, nxt_norm]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                line=dict(color=color, width=1.5), name=name, showlegend=False))
            min_w = t_max / 25
            for j in range(len(times) - 1):
                if times[j + 1] - times[j] >= min_w:
                    fig.add_annotation(x=(times[j] + times[j + 1]) / 2, y=y_mid,
                        text=f"h{values[j]:X}", showarrow=False,
                        font=dict(size=9, color=color, family="monospace"))
    fig.update_layout(
        xaxis=dict(title="Time", gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(tickvals=ticks, ticktext=labels, gridcolor="#21262d",
                   color="#8b949e", range=[-LANE_GAP, len(names) * LANE_TOTAL]),
        plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="monospace", size=11),
        margin=dict(l=110, r=20, t=20, b=50),
        height=max(280, len(names) * 52 + 80), hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# UI: CSS
# =============================================================================
def inject_css():
    st.markdown("""<style>
    [data-testid="stSidebar"] { border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] .stButton > button {
        text-align: left; font-family: monospace; font-size: 0.85rem; border-radius: 5px;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #21262d; border-radius: 8px; margin-bottom: 6px; padding: 4px 8px;
    }
    .stCode > div { border: 1px solid #30363d; border-radius: 6px; }
    [data-testid="stExpander"] { border: 1px solid #30363d !important; border-radius: 6px; }
    .stButton > button {
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        border-radius: 5px; transition: border-color 0.15s, color 0.15s;
    }
    hr { border-color: #30363d; }
    .block-container { padding-top: 1.2rem; }
    [data-testid="stChatInput"] textarea {
        font-family: 'Cascadia Code', 'Fira Code', monospace;
    }
    </style>""", unsafe_allow_html=True)


# =============================================================================
# UI: PAGES
# =============================================================================
def page_chat(db, llm_client, emb_model, theory_col, code_col, builtin_col):
    st.title("Verilog AI Assistant")
    _model_label = (f"OpenAI {LLM_MODEL_NAME}" if USE_OPENAI_API
                    else "Qwen2.5-Coder-32B (W4A16 SFT)")
    st.caption(f"{_model_label} — self-verifying via Icarus Verilog")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cooldown", f"{COOLDOWN_SECONDS}s / request")
    c2.metric("Max Prompt", f"{MAX_PROMPT_CHARS} chars")
    c3.metric("Correction", f"{MAX_CORRECTION_ATTEMPTS} attempts")
    st.divider()

    for msg in st.session_state.messages:
        render_message(msg)

    if query := st.chat_input("Enter your Verilog request..."):
        # Rate limit
        now = time.time()
        elapsed = now - st.session_state.last_message_time
        if elapsed < COOLDOWN_SECONDS:
            st.error(f"Please wait {int(COOLDOWN_SECONDS - elapsed)}s before sending another request.")
            st.stop()
        if len(query) > MAX_PROMPT_CHARS:
            st.error(f"Request too long ({len(query)} chars). Limit: {MAX_PROMPT_CHARS}.")
            st.stop()
        st.session_state.last_message_time = now

        # Append user message
        st.session_state.messages.append({"role": "user", "content": query})
        if st.session_state.active_chat_id is None:
            st.session_state.active_chat_id = str(uuid.uuid4())
            is_new = True
        else:
            is_new = False

        render_message({"role": "user", "content": query})

        # Math/off-topic bouncer — runs on EVERY turn.
        # Primary method: vector distance against theory corpus (precise).
        # Fallback: keyword check when theory corpus is unavailable.
        q_emb = get_embedding(query, emb_model)
        q_lower = query.lower().strip()
        bouncer_triggered = False

        if q_emb is not None and theory_col is not None:
            _, closest = retrieve_theory(theory_col, q_emb, k=1)
            if VERBOSE_DEBUG:
                print(f"[Bouncer] query={query!r} dist={closest:.4f}")
            if closest > MAX_DISTANCE_THRESHOLD:
                bouncer_triggered = True
        else:
            # Keyword-based fallback when theory corpus unavailable
            HDL_KEYWORDS = (
                'verilog', 'hdl', 'vhdl', 'systemverilog', 'rtl', 'module', 'testbench',
                'tb_', 'always', 'initial', 'wire', 'reg', 'logic', 'clock', 'clk',
                'reset', 'rst', 'flip-flop', 'flipflop', 'flop', 'latch', 'gate',
                'mux', 'multiplexer', 'decoder', 'encoder', 'adder', 'counter',
                'register', 'shift', 'fsm', 'state machine', 'fifo', 'ram', 'rom',
                'alu', 'comparator', 'lfsr', 'johnson', 'synthesis', 'posedge',
                'negedge', 'assign', 'endmodule', 'simulation', 'iverilog', 'vcd',
                'metastab', 'setup time', 'hold time', 'propagation', 'blocking',
                'non-blocking', 'nonblocking', 'combinational', 'sequential',
                'digital design', 'hardware description', 'bit', 'binary', 'nand',
                'nor ', 'xor', 'xnor', 'and gate', 'or gate', 'not gate', 'inverter',
            )
            OFFTOPIC_KEYWORDS = (
                'weather', 'joke', 'recipe', 'cooking', 'sport', 'movie', 'music',
                'news', 'politics', 'stock', 'crypto', 'bitcoin', 'dating',
            )
            has_hdl = any(kw in q_lower for kw in HDL_KEYWORDS)
            has_offtopic = any(kw in q_lower for kw in OFFTOPIC_KEYWORDS)
            # Trigger bouncer if explicitly off-topic, or very short query with no HDL signal
            if has_offtopic and not has_hdl:
                bouncer_triggered = True
            elif len(q_lower) < 25 and not has_hdl and not re.search(r'\d', q_lower):
                # Short queries with no HDL terms and no digits — probably chat noise
                bouncer_triggered = True

        if bouncer_triggered:
            bouncer_msg = ("I can only help with Verilog and HDL topics — "
                           "your question looks outside that scope.")
            with st.chat_message("assistant"):
                st.info(bouncer_msg)
            st.session_state.messages.append({"role": "assistant", "content": bouncer_msg})
            if st.session_state.user_info:
                save_messages(db, st.session_state.user_info['uid'],
                              st.session_state.active_chat_id, st.session_state.messages)
            st.rerun()
            return

        # Route: code vs concept
        code_mode = is_code_request(query)

        with st.chat_message("assistant"):
            container = st.empty()
            if code_mode:
                # Retrieve code examples + theory
                code_examples = retrieve_code_examples(code_col, builtin_col, emb_model, query)
                theory_docs = []
                if q_emb is not None:
                    theory_docs, _ = retrieve_theory(theory_col, q_emb, k=1)
                response = generate_code_response(
                    llm_client, query, code_examples, theory_docs, container
                )
            else:
                # Concept mode
                theory_docs = []
                if q_emb is not None:
                    theory_docs, _ = retrieve_theory(theory_col, q_emb, k=2)
                response = generate_concept_response(
                    llm_client, query, theory_docs, container
                )
            container.empty()
            render_message({"role": "assistant", "content": response})

        st.session_state.messages.append({"role": "assistant", "content": response})
        if st.session_state.user_info:
            save_messages(db, st.session_state.user_info['uid'],
                          st.session_state.active_chat_id, st.session_state.messages)
            if is_new:
                st.session_state.chat_sessions = get_chat_sessions(
                    db, st.session_state.user_info['uid'])
        st.rerun()


def page_simulator():
    st.title("Simulator Sandbox")
    _env = f"WSL ({WSL_DISTRO})" if USE_WSL else "Linux"
    st.caption(f"Icarus Verilog via {_env} · iverilog -g2012 · local execution")
    st.divider()

    _k = st.session_state.ace_key_counter
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Design Module")
        v = st_ace(value=st.session_state.verilog_code, language="verilog",
                   theme="tomorrow_night_blue", height=420,
                   key=f"design_{_k}", font_size=13, wrap=False)
        if v is not None:
            st.session_state.verilog_code = v
    with c2:
        st.subheader("Testbench")
        t = st_ace(value=st.session_state.testbench_code, language="verilog",
                   theme="tomorrow_night_blue", height=420,
                   key=f"tb_{_k}", font_size=13, wrap=False)
        if t is not None:
            st.session_state.testbench_code = t

    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Compiling and simulating..."):
            sim = run_simulation(st.session_state.verilog_code,
                                 st.session_state.testbench_code)

        r1, r2, r3 = st.columns(3)
        if sim["env_error"]:
            r1.metric("Compile", "N/A"); r2.metric("Simulation", "N/A"); r3.metric("Status", "ENV ERR")
        else:
            r1.metric("Compile", "PASS" if not sim["compile_error"] else "FAIL")
            r2.metric("Simulation", "PASS" if sim["success"] else
                      ("FAIL" if sim["functional_fail"] else "N/A"))
            r3.metric("Status", "OK" if sim["success"] else "ERROR")

        if sim["env_error"]:
            st.error("Environment error — not a Verilog bug.")
            st.code(sim["raw_errors"], language="text")
        elif sim["success"]:
            st.success("All stages passed.")
        elif sim["compile_error"]:
            st.error("Compilation error.")
            st.code(sim["raw_errors"], language="text")
        else:
            st.warning("Simulation ran but testbench reported failures.")
            if sim["raw_errors"]:
                st.code(sim["raw_errors"], language="text")

        with st.expander("Simulation Output", expanded=True):
            if sim["simulation_output"]:
                st.code(sim["simulation_output"], language="text")
            else:
                st.info("No terminal output produced.")
        with st.expander("Full Log"):
            st.code(sim["log"], language="text")

        st.divider()
        if sim.get("vcd_data"):
            dl, info = st.columns([1, 3])
            with dl:
                st.download_button("Download .vcd", data=sim["vcd_data"],
                                   file_name="waveform.vcd", mime="text/plain",
                                   use_container_width=True)
            with info:
                st.caption("Open in GTKWave or wavetrace.io for a full viewer. "
                           "Preview below.")
            with st.expander("Waveform Preview", expanded=True):
                render_waveform_chart(sim["vcd_data"])
        elif not sim["compile_error"]:
            st.info("No VCD generated. Add `$dumpfile(\"waveform.vcd\"); "
                    "$dumpvars(0, <tb>);` to your testbench's initial block.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Session state initialization
    defaults = {
        "user_info": None, "active_chat_id": None, "messages": [],
        "chat_sessions": [], "last_message_time": 0, "ace_key_counter": 0,
        "verilog_code": "module and_gate(input a, input b, output y);\n    assign y = a & b;\nendmodule\n",
        "testbench_code": "`timescale 1ns/1ps\nmodule tb_and_gate;\n    reg a, b;\n    wire y;\n    and_gate dut(.a(a),.b(b),.y(y));\n    initial begin\n        $dumpfile(\"waveform.vcd\"); $dumpvars(0, tb_and_gate);\n        a=0; b=0; #10; if (y !== 1'b0) $display(\"TB_FAIL\");\n        a=0; b=1; #10; if (y !== 1'b0) $display(\"TB_FAIL\");\n        a=1; b=0; #10; if (y !== 1'b0) $display(\"TB_FAIL\");\n        a=1; b=1; #10; if (y !== 1'b1) $display(\"TB_FAIL\");\n        $display(\"TB_PASS\"); $finish;\n    end\nendmodule\n",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Load resources
    res = initialize_resources()
    db            = res["firebase_db"]
    llm_client    = res["llm_client"]
    emb_model     = res["embedding_model"]
    theory_col    = res["theory_collection"]
    code_col      = res["code_rag_collection"]
    builtin_col   = res["builtin_collection"]

    if not VLLM_API_KEY:
        st.warning("VLLM_API_KEY not set. Code generation will fail until configured.", icon="⚠️")

    inject_css()

    # Sidebar
    with st.sidebar:
        st.title("Verilog AI")
        st.caption("OpenAI API · verification loop" if USE_OPENAI_API
                   else "Qwen2.5-Coder-32B · W4A16 SFT")
        st.divider()

        if st.button("+ New Chat", use_container_width=True, type="primary"):
            st.session_state.active_chat_id = None
            st.session_state.messages = []
            st.rerun()

        app_mode = st.radio("View", ("Chat", "Simulator Sandbox"),
                            label_visibility="collapsed", key="nav_radio")
        st.divider()

        with st.expander("Account", expanded=not st.session_state.user_info):
            if st.session_state.user_info:
                st.success(f"Signed in as {st.session_state.user_info.get('email')}")
                if st.button("Sign Out", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
            else:
                # Email + password auth via Firebase REST API (verifies password).
                # Falls back to admin-SDK email-only lookup if FIREBASE_WEB_API_KEY
                # not configured (useful for local dev).
                with st.form("login"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    cc1, cc2 = st.columns(2)
                    if cc1.form_submit_button("Login"):
                        if not email or not password:
                            st.error("Email and password required.")
                        elif FIREBASE_WEB_API_KEY:
                            # Real password verification
                            result = firebase_signin_password(email, password)
                            if result["ok"]:
                                st.session_state.user_info = {
                                    "uid": result["uid"], "email": result["email"]
                                }
                                st.rerun()
                            else:
                                st.error(result["error"])
                        else:
                            # Dev fallback: admin SDK email lookup (NO password check)
                            try:
                                user = auth.get_user_by_email(email)
                                st.session_state.user_info = {"uid": user.uid, "email": user.email}
                                st.warning("Dev mode: password not verified. "
                                           "Set FIREBASE_WEB_API_KEY for production.")
                                st.rerun()
                            except Exception:
                                st.error("Login failed — user not found.")
                    if cc2.form_submit_button("Sign Up"):
                        if not email or not password:
                            st.error("Email and password required.")
                        elif len(password) < 6:
                            st.error("Password must be 6+ characters.")
                        elif FIREBASE_WEB_API_KEY:
                            result = firebase_signup_password(email, password)
                            if result["ok"]:
                                st.session_state.user_info = {
                                    "uid": result["uid"], "email": result["email"]
                                }
                                st.rerun()
                            else:
                                st.error(result["error"])
                        else:
                            try:
                                user = auth.create_user(email=email, password=password)
                                st.session_state.user_info = {"uid": user.uid, "email": user.email}
                                st.rerun()
                            except Exception as e:
                                st.error(f"Sign up failed: {e}")

        if st.session_state.user_info:
            st.subheader("Chat History")
            if not st.session_state.chat_sessions:
                st.session_state.chat_sessions = get_chat_sessions(
                    db, st.session_state.user_info['uid'])
            for s in st.session_state.chat_sessions:
                cc1, cc2 = st.columns([5, 1])
                with cc1:
                    if st.button(s["title"], key=f"sel_{s['id']}", use_container_width=True):
                        st.session_state.active_chat_id = s["id"]
                        st.session_state.messages = load_messages(
                            db, st.session_state.user_info['uid'], s["id"])
                        st.rerun()
                with cc2:
                    if st.button("x", key=f"del_{s['id']}", use_container_width=True):
                        delete_chat(db, st.session_state.user_info['uid'], s["id"])
                        if st.session_state.active_chat_id == s["id"]:
                            st.session_state.active_chat_id = None
                            st.session_state.messages = []
                        st.session_state.chat_sessions = get_chat_sessions(
                            db, st.session_state.user_info['uid'])
                        st.rerun()

    # Main content
    if not st.session_state.user_info:
        st.title("Verilog AI Platform")
        st.markdown("A self-verifying Verilog assistant powered by a fine-tuned "
                    "Qwen2.5-Coder-32B model. Sign in to get started.")
    else:
        if app_mode == "Chat":
            page_chat(db, llm_client, emb_model, theory_col, code_col, builtin_col)
        else:
            page_simulator()


if __name__ == "__main__":
    main()