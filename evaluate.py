"""
HDL Sentinel — Evaluation Harness
===================================
Runs a fixed benchmark of 28 Verilog design prompts through the full generation
pipeline and measures simulation pass rates.

Configurations tested (ablation study):
  A. Full pipeline  — up to 3 correction retries
  B. No corrections — single generation attempt only
  C. 1 correction   — one retry allowed

Usage:
  python evaluate.py              # full eval (~84 LLM calls, ~20-30 min)
  python evaluate.py --quick      # Config A only, first 12 prompts (~10 min)

Requires:
  OPENAI_API_KEY env var  (PowerShell: $env:OPENAI_API_KEY = "sk-...")
  iverilog in WSL Ubuntu-22.04 (Windows) or system PATH (Linux/Mac)

Outputs:
  eval_results.json   — full raw results
  eval_results.csv    — one row per prompt per config
  eval_charts.png     — 4-panel presentation figure
"""

import os, re, sys, json, csv, time, subprocess, tempfile, argparse
from collections import defaultdict

# ── Load .env file if present (fallback for env vars) ───────────────────────
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ── Required deps ────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# RAG deps — optional, graceful fallback
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — charts will be skipped. pip install matplotlib")

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "").strip()
LLM_MODEL        = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

CODE_RAG_DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db_code_examples")
EMB_MODEL_PATH   = os.path.join(SCRIPT_DIR, "embedding_model")
CODE_RAG_COLLECTION = "verilog_code_examples"

MAX_CORRECTION_ATTEMPTS = 3
MAX_COMPLETION_TOKENS   = 2048
SIMULATION_TIMEOUT      = 30
WSL_DISTRO              = "Ubuntu-22.04"
CODE_RAG_TOP_K          = 2
CODE_RAG_MAX_DIST       = 0.45

USE_WSL = sys.platform == "win32"

# =============================================================================
# BENCHMARK — 28 prompts across all design types
# =============================================================================
BENCHMARK = [
    # ── Gates (pure combinational) ──────────────────────────────────────────
    {"id": "gate_and",        "prompt": "design a 2-input AND gate",                                                                   "type": "gate"},
    {"id": "gate_or",         "prompt": "design a 2-input OR gate",                                                                    "type": "gate"},
    {"id": "gate_xor",        "prompt": "design a 2-input XOR gate",                                                                   "type": "gate"},
    {"id": "gate_nand",       "prompt": "design a 2-input NAND gate",                                                                  "type": "gate"},
    # ── Combinational ───────────────────────────────────────────────────────
    {"id": "mux_2to1",        "prompt": "design a 2-to-1 multiplexer",                                                                 "type": "mux"},
    {"id": "mux_4to1",        "prompt": "design a 4-to-1 multiplexer with 2-bit select",                                               "type": "mux"},
    {"id": "decoder_2to4",    "prompt": "design a 2-to-4 decoder with enable",                                                         "type": "decoder"},
    {"id": "encoder_4to2",    "prompt": "design a 4-to-2 priority encoder",                                                            "type": "encoder"},
    {"id": "half_adder",      "prompt": "design a half adder",                                                                         "type": "adder"},
    {"id": "full_adder",      "prompt": "design a full adder with carry in and carry out",                                             "type": "adder"},
    {"id": "adder_4bit",      "prompt": "design a 4-bit ripple carry adder",                                                           "type": "adder"},
    {"id": "comparator_4bit", "prompt": "design a 4-bit magnitude comparator with equal, greater-than, and less-than outputs",         "type": "combinational"},
    # ── Flip-flops ──────────────────────────────────────────────────────────
    {"id": "dff_sync",        "prompt": "design a D flip-flop with synchronous reset",                                                 "type": "dff"},
    {"id": "tff",             "prompt": "design a T flip-flop with synchronous reset",                                                 "type": "tff"},
    {"id": "jkff",            "prompt": "design a JK flip-flop with synchronous reset",                                               "type": "jkff"},
    # ── Counters ────────────────────────────────────────────────────────────
    {"id": "counter_4bit",    "prompt": "design a 4-bit binary up counter with enable and synchronous reset",                          "type": "counter"},
    {"id": "counter_down3",   "prompt": "design a 3-bit binary down counter with synchronous reset",                                   "type": "counter"},
    {"id": "counter_mod6",    "prompt": "design a modulo-6 counter with synchronous reset that counts 0 to 5",                        "type": "counter"},
    {"id": "counter_bcd",     "prompt": "design a BCD counter that counts from 0 to 9 and resets with synchronous reset",             "type": "counter"},
    # ── Shift Registers ─────────────────────────────────────────────────────
    {"id": "shift_sipo",      "prompt": "design a 4-bit serial-in parallel-out shift register with synchronous reset",                 "type": "shift_reg"},
    {"id": "shift_piso",      "prompt": "design an 8-bit parallel-in serial-out shift register with load and shift control",          "type": "shift_reg"},
    # ── FSM ─────────────────────────────────────────────────────────────────
    {"id": "fsm_traffic",     "prompt": "design a traffic light FSM with three states: red, green, yellow, each lasting 3 clock cycles", "type": "fsm"},
    {"id": "fsm_seq_det",     "prompt": "design a Mealy sequence detector FSM that detects the bit sequence 1011",                    "type": "fsm"},
    # ── ALU ─────────────────────────────────────────────────────────────────
    {"id": "alu_4bit",        "prompt": "design a 4-bit ALU with 2-bit opcode supporting add, subtract, AND, OR operations",          "type": "alu"},
    # ── Special ─────────────────────────────────────────────────────────────
    {"id": "lfsr_4bit",       "prompt": "design a 4-bit Fibonacci LFSR with synchronous reset and taps at bits 3 and 2",              "type": "lfsr"},
    {"id": "johnson_4bit",    "prompt": "design a 4-bit Johnson counter with synchronous reset",                                       "type": "johnson"},
    # ── Complex ─────────────────────────────────────────────────────────────
    {"id": "fifo_sync",       "prompt": "design a synchronous FIFO with 8 entries and 8-bit data width with full and empty flags",    "type": "fifo"},
    {"id": "uart_tx",         "prompt": "design a UART transmitter with 8 data bits, 1 stop bit, no parity, and a parameter for clock divider", "type": "uart"},
]

# =============================================================================
# SYSTEM PROMPT (same as app.py)
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
    "ABSOLUTE RULES:\n"
    "- DUT instantiation syntax: `module_name dut (.port1(sig1), .port2(sig2));` "
    "— ONE opening paren after `dut`. NEVER write `.b(b)), .y(y))` — extra `)` is a syntax error.\n"
    "- Design and testbench port names MUST match exactly.\n"
    "- Do NOT use undeclared identifiers like WIDTH, CLK_PERIOD — use hardcoded literals.\n"
    "- Testbench MUST include `$dumpfile(\"waveform.vcd\"); $dumpvars(0, <tb_module_name>);`\n"
    "- End testbench with `$display(\"TB_PASS\"); $finish;` after all checks pass.\n"
    "- Include `initial #20000 begin $display(\"TB_FAIL: timeout\"); $finish; end` as safety.\n"
    "- Use ONLY `if (signal !== expected) begin $display(\"TB_FAIL: ...\"); $finish; end` for assertions.\n"
    "- DO NOT use SystemVerilog `assert(...)`, `$fatal`, `$error`, or SVA constructs.\n"
    "- Use `reg` for testbench signals (not `logic`) for maximum iverilog compatibility.\n"
    "- For simple combinational circuits, use a single `assign` statement. No clock needed.\n"
    "- Timing: sequential → `@(posedge clk); #1;` before checks. Combinational → `#10; #1;`\n"
)

CORRECTION_SYSTEM_PROMPT = (
    "You are fixing a specific Verilog error. Output the corrected design module "
    "and testbench in two separate ```verilog code blocks. "
    "Fix only what the error describes. Keep port names consistent between design and testbench."
)

# =============================================================================
# RAG INITIALIZATION
# =============================================================================
_rag_code_col  = None
_rag_emb_model = None

def init_rag():
    global _rag_code_col, _rag_emb_model
    if not HAS_RAG_DEPS:
        return
    # Embedding model
    if os.path.isdir(EMB_MODEL_PATH):
        try:
            _rag_emb_model = SentenceTransformer(EMB_MODEL_PATH)
            print(f"[RAG] Embedding model loaded from: {EMB_MODEL_PATH}")
        except Exception as e:
            print(f"[RAG] Embedding model load failed: {e}")
    else:
        try:
            _rag_emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("[RAG] Embedding model loaded from HuggingFace Hub")
        except Exception as e:
            print(f"[RAG] Embedding model load failed: {e}")
    # Code corpus
    if os.path.isdir(CODE_RAG_DB_PATH) and _rag_emb_model is not None:
        try:
            cli = chromadb.PersistentClient(path=CODE_RAG_DB_PATH)
            _rag_code_col = cli.get_collection(name=CODE_RAG_COLLECTION)
            print(f"[RAG] Code corpus loaded: {_rag_code_col.count():,} documents")
        except Exception as e:
            print(f"[RAG] Code corpus unavailable: {e}")
    else:
        print(f"[RAG] Code corpus not found at {CODE_RAG_DB_PATH} — RAG disabled")


def retrieve_code_examples(query: str, k: int = CODE_RAG_TOP_K) -> list:
    if _rag_code_col is None or _rag_emb_model is None:
        return []
    try:
        q_emb = _rag_emb_model.encode([query], normalize_embeddings=True).tolist()
        res   = _rag_code_col.query(query_embeddings=q_emb, n_results=k,
                                    include=["documents", "distances"])
        docs  = res.get("documents", [[]])[0]
        dists = res.get("distances",  [[]])[0]
        return [d for d, dist in zip(docs, dists) if dist < CODE_RAG_MAX_DIST]
    except Exception as e:
        print(f"[RAG] query failed: {e}")
        return []


# =============================================================================
# SIMULATION  (mirrors app.py exactly)
# =============================================================================
FAIL_MARKERS = ["TB_FAIL", "MISMATCH", "ASSERTION FAILED"]
MAX_ERROR_CHARS = 600


def win_to_wsl_path(windows_path: str) -> str:
    p = os.path.abspath(windows_path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return p


def run_simulation(design_code: str, tb_code: str) -> dict:
    result = {"success": False, "compile_error": False, "functional_fail": False,
              "env_error": False, "raw_errors": ""}
    if not design_code.strip() or not tb_code.strip():
        result["env_error"] = True
        result["raw_errors"] = "Empty code"
        return result

    with tempfile.TemporaryDirectory() as tmp:
        design_path = os.path.join(tmp, "design.v")
        tb_path     = os.path.join(tmp, "tb_design.v")
        vvp_path    = os.path.join(tmp, "simulation.vvp")

        with open(design_path, "w", encoding="utf-8") as f: f.write(design_code)
        with open(tb_path,     "w", encoding="utf-8") as f: f.write(tb_code)

        if USE_WSL:
            d_arg   = win_to_wsl_path(design_path)
            tb_arg  = win_to_wsl_path(tb_path)
            vvp_arg = win_to_wsl_path(vvp_path)
            tmp_arg = win_to_wsl_path(tmp)
            compile_cmd = ["wsl", "-d", WSL_DISTRO, "iverilog", "-g2012", "-o", vvp_arg, d_arg, tb_arg]
            sim_cmd     = ["wsl", "-d", WSL_DISTRO, "sh", "-c", f"cd '{tmp_arg}' && vvp '{vvp_arg}'"]
        else:
            compile_cmd = ["iverilog", "-g2012", "-o", vvp_path, design_path, tb_path]
            sim_cmd     = ["sh", "-c", f"cd '{tmp}' && vvp '{vvp_path}'"]

        try:
            proc = subprocess.run(compile_cmd, capture_output=True, text=True,
                                  timeout=SIMULATION_TIMEOUT, encoding="utf-8")
            if proc.returncode != 0:
                result["compile_error"] = True
                result["raw_errors"] = (proc.stderr or proc.stdout)[:MAX_ERROR_CHARS]
                return result
        except FileNotFoundError:
            result["env_error"] = True
            result["raw_errors"] = "iverilog not found"
            return result
        except subprocess.TimeoutExpired:
            result["env_error"] = True
            result["raw_errors"] = "compile timeout"
            return result

        try:
            proc = subprocess.run(sim_cmd, capture_output=True, text=True,
                                  timeout=SIMULATION_TIMEOUT, encoding="utf-8")
            out = proc.stdout
            fails = [l for l in out.splitlines() if any(m.lower() in l.lower() for m in FAIL_MARKERS)]
            if fails:
                result["functional_fail"] = True
                result["raw_errors"] = "\n".join(fails[:3])[:MAX_ERROR_CHARS]
            elif "TB_PASS" in out.upper():
                result["success"] = True
            elif proc.returncode == 0 and out.strip():
                result["success"] = True
            else:
                result["functional_fail"] = True
                result["raw_errors"] = "\n".join(out.splitlines()[-5:])[:MAX_ERROR_CHARS]
        except subprocess.TimeoutExpired:
            result["env_error"] = True
            result["raw_errors"] = "simulation timeout"

    return result


# =============================================================================
# CODE EXTRACTION  (mirrors app.py)
# =============================================================================
def looks_like_tb(code: str) -> bool:
    if not code:
        return False
    has_initial = bool(re.search(r'\binitial\b', code))
    has_display = bool(re.search(r'\$display|\$finish|\$dumpfile', code))
    has_dut     = bool(re.search(r'\b(?:dut|uut)\s*\(', code, re.IGNORECASE))
    has_tb_name = bool(re.search(r'\bmodule\s+(?:tb_|\w*_?tb\b)', code, re.IGNORECASE))
    return has_tb_name or (has_initial and has_display and has_dut)


def extract_code_blocks(text: str):
    if not text:
        return "", ""
    blocks = []
    for b in re.findall(r'```verilog\s*\n(.*?)```', text, re.DOTALL):
        bs = b.strip()
        if 'module' in bs and 'endmodule' in bs:
            blocks.append(bs)
    if not blocks:
        for b in re.findall(r'```[^\n]*\n(.*?)```', text, re.DOTALL):
            bs = b.strip()
            if 'module' in bs and 'endmodule' in bs:
                blocks.append(bs)
    if not blocks:
        return "", ""

    designs = [b for b in blocks if not looks_like_tb(b)]
    tbs     = [b for b in blocks if looks_like_tb(b)]
    design  = designs[0] if designs else (blocks[0] if blocks else "")
    tb      = tbs[0]     if tbs     else (blocks[1] if len(blocks) >= 2 else "")
    return design, tb


def is_complete_module(code: str) -> bool:
    if not code or not code.strip():
        return False
    if not re.search(r'\bmodule\s+\w+\s*[\(;]', code):
        return False
    if 'endmodule' not in code:
        return False
    return True


def ensure_vcd_dump(tb_code: str) -> str:
    if '$dumpfile' in tb_code:
        return tb_code
    m = re.search(r'\bmodule\s+(\w+)', tb_code)
    mod_name = m.group(1) if m else "tb"
    return re.sub(r'(\binitial\b\s+begin\b)',
                  f'\\1\n        $dumpfile("waveform.vcd");\n        $dumpvars(0, {mod_name});',
                  tb_code, count=1)


# =============================================================================
# LLM CALL
# =============================================================================
def call_llm(client, messages: list, max_tokens: int = MAX_COMPLETION_TOKENS) -> str | None:
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=0.1, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                wait = 2 ** (attempt + 1)
                print(f"  [rate-limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [API error] {e}")
            return None
    return None


# =============================================================================
# GENERATION + EVALUATION
# =============================================================================
def evaluate_prompt(client, prompt: str, use_rag: bool, max_corrections: int) -> dict:
    """
    Run one benchmark prompt through the generation pipeline.
    Returns a result dict with pass_attempt (1-indexed, 0 = failed all).
    """
    code_examples = retrieve_code_examples(prompt) if use_rag else []
    context = ""
    if code_examples:
        context = "[Reference code examples — match this style]\n" + "\n\n---\n\n".join(code_examples) + "\n\n"

    user_content = f"{context}Request: {prompt}"
    messages = [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    raw = call_llm(client, messages)
    if raw is None:
        return {"outcome": "api_error", "pass_attempt": 0, "attempts": 0,
                "compile_errors": 0, "functional_fails": 0}

    design, tb = extract_code_blocks(raw)
    if not is_complete_module(design) or not is_complete_module(tb):
        return {"outcome": "extraction_fail", "pass_attempt": 0, "attempts": 1,
                "compile_errors": 0, "functional_fails": 0}

    tb = ensure_vcd_dump(tb)

    compile_errors = 0
    functional_fails = 0

    for attempt in range(max_corrections + 1):
        sim = run_simulation(design, tb)

        if sim["env_error"]:
            return {"outcome": "env_error", "pass_attempt": 0, "attempts": attempt + 1,
                    "compile_errors": compile_errors, "functional_fails": functional_fails}

        if sim["success"]:
            return {"outcome": "pass", "pass_attempt": attempt + 1,
                    "attempts": attempt + 1,
                    "compile_errors": compile_errors,
                    "functional_fails": functional_fails}

        if sim["compile_error"]:
            compile_errors += 1
        else:
            functional_fails += 1

        if attempt >= max_corrections:
            break

        # Correction turn
        err_kind = "compile error" if sim["compile_error"] else "simulation failure"
        correction = (
            f"Your previous design had a {err_kind}:\n"
            f"```\n{sim['raw_errors']}\n```\n"
            f"Output the corrected design module AND testbench in two separate "
            f"```verilog code blocks. Keep port names consistent. Fix only what the error describes."
        )
        messages_corr = [
            {"role": "system",    "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": raw},
            {"role": "user",      "content": correction},
        ]
        raw2 = call_llm(client, messages_corr)
        if raw2 is None:
            break
        new_design, new_tb = extract_code_blocks(raw2)
        if is_complete_module(new_design) and is_complete_module(new_tb):
            design = new_design
            tb     = ensure_vcd_dump(new_tb)
            raw    = raw2

    return {"outcome": "fail", "pass_attempt": 0,
            "attempts": max_corrections + 1,
            "compile_errors": compile_errors,
            "functional_fails": functional_fails}


# =============================================================================
# RUN BENCHMARK
# =============================================================================
CONFIGS = [
    {"label": "Full Pipeline\n(3 corrections)",  "key": "full",    "use_rag": HAS_RAG_DEPS, "max_corrections": 3},
    {"label": "No Corrections\n(1 attempt only)", "key": "no_corr", "use_rag": HAS_RAG_DEPS, "max_corrections": 0},
    {"label": "1 Correction\n(2 attempts)",       "key": "one_corr","use_rag": HAS_RAG_DEPS, "max_corrections": 1},
]


def run_benchmark(client, prompts: list, configs: list) -> list:
    all_results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: {cfg['label'].replace(chr(10), ' ')}")
        print(f"  use_rag={cfg['use_rag']}  max_corrections={cfg['max_corrections']}")
        print('='*60)
        for item in prompts:
            print(f"  [{item['id']:20s}] ", end="", flush=True)
            t0 = time.time()
            res = evaluate_prompt(client, item["prompt"], cfg["use_rag"], cfg["max_corrections"])
            elapsed = time.time() - t0
            outcome_str = (f"PASS@{res['pass_attempt']}" if res["outcome"] == "pass"
                           else res["outcome"].upper())
            print(f"{outcome_str:12s}  ({elapsed:.0f}s)")
            all_results.append({
                "config":          cfg["key"],
                "config_label":    cfg["label"].replace("\n", " "),
                "id":              item["id"],
                "prompt":          item["prompt"],
                "type":            item["type"],
                "outcome":         res["outcome"],
                "pass_attempt":    res["pass_attempt"],
                "attempts":        res["attempts"],
                "compile_errors":  res["compile_errors"],
                "functional_fails":res["functional_fails"],
            })
            # Short pause to respect rate limits
            time.sleep(1)
    return all_results


# =============================================================================
# CHARTS
# =============================================================================
PALETTE = {
    "full":     "#2ea44f",   # green
    "no_corr":  "#f85149",   # red
    "one_corr": "#e36209",   # orange
    "pass1":    "#2ea44f",
    "pass2":    "#56d364",
    "pass3":    "#94d82d",
    "fail":     "#f85149",
}

ATTEMPT_COLORS = ["#2ea44f", "#56d364", "#94d82d", "#f85149"]   # pass@1, @2, @3, fail


def compute_stats(results: list, config_key: str):
    rows = [r for r in results if r["config"] == config_key]
    n = len(rows)
    if n == 0:
        return {}
    passes     = [r for r in rows if r["outcome"] == "pass"]
    pass_rate  = len(passes) / n * 100
    pass1      = sum(1 for r in rows if r["pass_attempt"] == 1)
    pass2      = sum(1 for r in rows if r["pass_attempt"] == 2)
    pass3      = sum(1 for r in rows if r["pass_attempt"] == 3)
    pass4      = sum(1 for r in rows if r["pass_attempt"] >= 4)   # 4th attempt = 3rd correction
    fail_total = sum(1 for r in rows if r["outcome"] != "pass")
    return {
        "pass_rate": pass_rate,
        "pass1_pct": pass1 / n * 100,
        "pass2_pct": pass2 / n * 100,
        "pass3_pct": pass3 / n * 100,
        "pass4_pct": pass4 / n * 100,
        "fail_pct":  fail_total / n * 100,
        "n": n,
    }


def make_charts(results: list, out_path: str):
    if not HAS_MPL:
        print("[charts] matplotlib not available — skipping chart generation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    ax_style = dict(facecolor="#161b22", labelcolor="#c9d1d9", titlecolor="#e6edf3")

    for ax in axes.flat:
        ax.set_facecolor(ax_style["facecolor"])
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

    # ── Chart 1: Overall pass rate per config ───────────────────────────────
    ax1 = axes[0, 0]
    cfg_keys    = [c["key"]   for c in CONFIGS]
    cfg_labels  = [c["label"] for c in CONFIGS]
    pass_rates  = [compute_stats(results, k).get("pass_rate", 0) for k in cfg_keys]
    colors      = [PALETTE[k] for k in cfg_keys]
    bars = ax1.bar(range(len(cfg_keys)), pass_rates, color=colors, width=0.5,
                   edgecolor="#30363d", linewidth=0.8)
    ax1.set_xticks(range(len(cfg_keys)))
    ax1.set_xticklabels(cfg_labels, fontsize=9, color="#c9d1d9")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Pass Rate (%)", color="#c9d1d9", fontsize=10)
    ax1.set_title("Overall Pass Rate — Ablation Study", color="#e6edf3", fontsize=11, pad=10)
    for bar, rate in zip(bars, pass_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{rate:.0f}%", ha="center", va="bottom", color="#e6edf3", fontsize=11, fontweight="bold")
    ax1.axhline(y=100, color="#30363d", linestyle="--", linewidth=0.5)

    # ── Chart 2: Pass@1 / Pass@2 / Pass@3 / Pass@4 / Fail stacked ──────────
    ax2 = axes[0, 1]
    full_stats  = compute_stats(results, "full")
    cats        = ["Pass@1", "Pass@2", "Pass@3", "Pass@4\n(3rd correction)", "Fail"]
    vals        = [full_stats.get("pass1_pct", 0), full_stats.get("pass2_pct", 0),
                   full_stats.get("pass3_pct", 0), full_stats.get("pass4_pct", 0),
                   full_stats.get("fail_pct",  0)]
    colors5     = ["#2ea44f", "#56d364", "#94d82d", "#d2a600", "#f85149"]
    bottom = 0
    for cat, val, color in zip(cats, vals, colors5):
        if val > 0:
            ax2.bar([0], [val], bottom=bottom, color=color, width=0.4,
                    edgecolor="#30363d", linewidth=0.8, label=cat)
            if val >= 5:
                ax2.text(0, bottom + val/2, f"{val:.0f}%",
                         ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            bottom += val
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0, 110)
    ax2.set_xticks([])
    ax2.set_ylabel("% of Prompts", color="#c9d1d9", fontsize=10)
    ax2.set_title("Full Pipeline — Breakdown by Attempt", color="#e6edf3", fontsize=11, pad=10)
    ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9)

    # ── Chart 3: Pass rate by design type (Full System) ─────────────────────
    ax3 = axes[1, 0]
    full_rows = [r for r in results if r["config"] == "full"]
    type_groups = defaultdict(list)
    for r in full_rows:
        type_groups[r["type"]].append(r["outcome"] == "pass")
    type_labels   = sorted(type_groups.keys())
    type_pass_pct = [sum(type_groups[t]) / len(type_groups[t]) * 100 for t in type_labels]
    colors3 = ["#2ea44f" if p == 100 else ("#e36209" if p >= 50 else "#f85149") for p in type_pass_pct]
    bars3 = ax3.barh(type_labels, type_pass_pct, color=colors3, edgecolor="#30363d", linewidth=0.8)
    ax3.set_xlim(0, 120)
    ax3.set_xlabel("Pass Rate (%)", color="#c9d1d9", fontsize=10)
    ax3.set_title("Pass Rate by Design Type (Full System)", color="#e6edf3", fontsize=11, pad=10)
    ax3.tick_params(axis="y", labelsize=9, colors="#c9d1d9")
    for bar, pct in zip(bars3, type_pass_pct):
        ax3.text(pct + 1, bar.get_y() + bar.get_height()/2,
                 f"{pct:.0f}%", va="center", color="#e6edf3", fontsize=9)
    ax3.axvline(x=100, color="#30363d", linestyle="--", linewidth=0.5)

    # ── Chart 4: Correction loop impact — per-type, Full vs No-Corrections ──
    ax4 = axes[1, 1]
    nocorr_rows = [r for r in results if r["config"] == "no_corr"]
    nocorr_groups = defaultdict(list)
    for r in nocorr_rows:
        nocorr_groups[r["type"]].append(r["outcome"] == "pass")

    common_types = sorted(set(type_groups.keys()) & set(nocorr_groups.keys()))
    full_pcts   = [sum(type_groups[t])   / len(type_groups[t])   * 100 for t in common_types]
    nocorr_pcts = [sum(nocorr_groups[t]) / len(nocorr_groups[t]) * 100 for t in common_types]
    x4 = range(len(common_types))
    w4 = 0.35
    ax4.bar([i - w4/2 for i in x4], full_pcts,   width=w4, color=PALETTE["full"],    label="3 Corrections",  edgecolor="#30363d")
    ax4.bar([i + w4/2 for i in x4], nocorr_pcts, width=w4, color=PALETTE["no_corr"], label="No Corrections", edgecolor="#30363d")
    ax4.set_xticks(list(x4))
    ax4.set_xticklabels(common_types, rotation=35, ha="right", fontsize=8, color="#c9d1d9")
    ax4.set_ylim(0, 120)
    ax4.set_ylabel("Pass Rate (%)", color="#c9d1d9", fontsize=10)
    ax4.set_title("Correction Loop Impact — 3 Corrections vs None", color="#e6edf3", fontsize=11, pad=10)
    ax4.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    ax4.axhline(y=100, color="#30363d", linestyle="--", linewidth=0.5)

    fig.suptitle("HDL Sentinel — Evaluation Results", color="#e6edf3",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[charts] Saved → {out_path}")
    plt.close(fig)


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_results(results: list, json_path: str, csv_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[save] JSON → {json_path}")

    fields = ["config", "config_label", "id", "type", "prompt",
              "outcome", "pass_attempt", "attempts", "compile_errors", "functional_fails"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"[save] CSV  → {csv_path}")


def print_summary(results: list):
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for cfg in CONFIGS:
        stats = compute_stats(results, cfg["key"])
        if not stats:
            continue
        label = cfg["label"].replace("\n", " ")
        print(f"\n{label}")
        print(f"  Pass Rate : {stats['pass_rate']:.1f}%  ({int(stats['pass_rate']*stats['n']/100)}/{stats['n']} prompts)")
        print(f"  Pass@1    : {stats['pass1_pct']:.1f}%")
        print(f"  Pass@2    : {stats['pass2_pct']:.1f}%")
        print(f"  Pass@3    : {stats['pass3_pct']:.1f}%")
        print(f"  Pass@4    : {stats['pass4_pct']:.1f}%")
        print(f"  Fail      : {stats['fail_pct']:.1f}%")
    print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HDL Sentinel Evaluation Harness")
    parser.add_argument("--quick", action="store_true",
                        help="Run Config A only on first 12 prompts (fast smoke test)")
    parser.add_argument("--config", choices=["full", "no_corr", "one_corr"],
                        help="Run only one specific config")
    parser.add_argument("--out-dir", default=SCRIPT_DIR,
                        help="Directory to write output files")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("  PowerShell : $env:OPENAI_API_KEY = 'sk-...'")
        print("  Or create a .env file in this directory with: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"HDL Sentinel — Evaluation Harness")
    print(f"  Model   : {LLM_MODEL}")
    print(f"  Platform: {'Windows + WSL ' + WSL_DISTRO if USE_WSL else 'Linux (native iverilog)'}")

    # Initialize RAG
    init_rag()
    rag_available = (_rag_code_col is not None)
    print(f"  RAG     : {'ENABLED (' + str(_rag_code_col.count()) + ' docs)' if rag_available else 'DISABLED (no corpus)'}")

    # Select prompts and configs
    prompts = BENCHMARK[:12] if args.quick else BENCHMARK
    configs = CONFIGS
    if args.quick:
        print(f"  Mode    : QUICK (12 prompts × 3 configs = 36 LLM calls, ~15 min)")
    elif args.config:
        configs = [c for c in CONFIGS if c["key"] == args.config]
        print(f"  Mode    : single config '{args.config}'")
    else:
        print(f"  Mode    : FULL ({len(prompts)} prompts × {len(configs)} configs = {len(prompts)*len(configs)} LLM calls)")

    input(f"\nPress Enter to start evaluation...")

    results = run_benchmark(client, prompts, configs)

    out_dir = args.out_dir
    json_path  = os.path.join(out_dir, "eval_results.json")
    csv_path   = os.path.join(out_dir, "eval_results.csv")
    chart_path = os.path.join(out_dir, "eval_charts.png")

    save_results(results, json_path, csv_path)
    cfg_keys_run = {r["config"] for r in results}
    if len(cfg_keys_run) >= 2:
        make_charts(results, chart_path)
    else:
        print("[charts] Skipped — run with all 3 configs (or use --quick / omit --config) for charts")

    print_summary(results)
    print("Done.")


if __name__ == "__main__":
    main()
