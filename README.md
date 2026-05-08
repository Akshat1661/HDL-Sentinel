---
title: HDL Sentinel — Verilog AI Assistant
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
short_description: Self-verifying Verilog assistant with iverilog + RAG
---

# HDL-Sentinel: Verilog AI Assistant

A self-verifying Verilog RTL design tutor for CSUF EGEC students.

**Architecture:**
- **LLM:** OpenAI gpt-4o-mini
- **RAG:** 14,740 curated Verilog examples + 16 built-in designs + PDF theory corpus
- **Verification:** Icarus Verilog compilation + simulation with 3-attempt correction loop
- **Auth:** Firebase Authentication (email + password)
- **History:** Per-user Firestore-backed chat persistence

**Required environment variables (set in HF Spaces → Settings → Variables and secrets):**
- `OPENAI_API_KEY` (secret)
- `FIREBASE_SERVICE_ACCOUNT_JSON` (secret — full JSON as string)
- `FIREBASE_WEB_API_KEY` (secret — Firebase Web API key for password auth)

**Optional:**
- `LLM_MODEL_NAME` (default: `gpt-4o-mini`)
- `VERBOSE_DEBUG` (set to `1` to enable debug prints)

## Author

Akshat Desai · M.S. Computer Science · California State University, Fullerton
Advisors: Dr. Rakesh Mahto · Dr. Kiran George · Dr. Kenneth John Faller II
