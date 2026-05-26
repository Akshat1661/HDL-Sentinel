# HDL Sentinel — Verilog AI Assistant

HDL Sentinel is a self-verifying Verilog RTL design tutor built for M.S. Computer Science students at California State University, Fullerton.

It generates synthesizable Verilog modules and testbenches, compiles and simulates them with Icarus Verilog, and uses simulator feedback to correct generation errors. The goal is to close the loop between LLM-generated code and ground-truth hardware simulation.

**Master's Thesis Project**  
Akshat Desai · M.S. Computer Science · California State University, Fullerton  
Advisors: Dr. Rakesh Mahto · Dr. Kiran George · Dr. Kenneth John Faller II

---

## Features

- AI code generation
- Self-correcting simulation loop
- RAG-augmented generation
- Dual-mode responses
- Simulator sandbox
- Waveform visualization
- Firebase authentication
- Rate limiting

---

## System Overview

The system is built around four components:

- **Streamlit frontend** for chat, code generation, and simulator sandbox workflows
- **RAG retrieval layer** using ChromaDB and sentence-transformer embeddings
- **LLM backend** using OpenAI API by default, with optional vLLM support
- **Icarus Verilog verification loop** using `iverilog` and `vvp` as the simulation oracle

The LLM is not treated as the final authority. Generated designs are compiled and simulated before being marked as verified.

---

## Key Design Decisions

- Icarus Verilog acts as the ground-truth oracle for compile and simulation validation.
- Correction prompts include compiler and simulator error logs so the model can perform targeted fixes.
- Concept questions and code-generation prompts use different retrieval and response paths.
- Generated vector databases, embedding models, credentials, and large datasets are excluded from the repository.

---

## Tech Stack

| Layer           | Technology                                              |
| --------------- | ------------------------------------------------------- |
| Frontend        | Streamlit, streamlit-ace                                |
| LLM             | OpenAI gpt-4o-mini, optional Qwen2.5-Coder-32B via vLLM |
| Retrieval       | ChromaDB, sentence-transformers/all-MiniLM-L6-v2        |
| Simulation      | Icarus Verilog, iverilog, vvp                           |
| Auth + Database | Firebase Authentication, Cloud Firestore                |
| Waveforms       | VCD parsing, Plotly                                     |
| Deployment      | Hugging Face Spaces, Streamlit runtime                  |

---

## Repository Structure

```
HDL-Sentinel/
├── app.py
├── evaluate.py
├── make_charts.py
├── requirements.txt
├── packages.txt
├── DEMO.md
├── eval_results.json
├── eval_results.csv
├── eval_charts.png
├── arch_slide.png
├── hdl_sentinel_architecture.svg
├── README.md
└── .streamlit/
```

---

## Setup

### Prerequisites

- Python 3.10+
- Icarus Verilog installed and available on PATH
- Firebase project with Authentication and Firestore enabled
- OpenAI API key or a running vLLM backend

### Install

```bash
git clone https://github.com/Akshat1661/HDL-Sentinel.git
cd HDL-Sentinel
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key
FIREBASE_WEB_API_KEY=your_firebase_web_api_key
FIREBASE_SERVICE_ACCOUNT_JSON=your_firebase_service_account_json

LLM_MODEL_NAME=gpt-4o-mini
VLLM_BASE_URL=http://localhost:8000
VLLM_API_KEY=your_vllm_key
VERBOSE_DEBUG=1
```

> **Warning:** Do not commit `.env`, Firebase service account files, API keys, generated vector databases, or embedding models.

### Run Locally

```bash
streamlit run app.py
```

---

## Deployment

HDL Sentinel is deployed on Hugging Face Spaces using the Streamlit runtime.

**Required secrets** (set under Settings → Variables and secrets):

- `OPENAI_API_KEY`
- `FIREBASE_WEB_API_KEY`
- `FIREBASE_SERVICE_ACCOUNT_JSON`

**Optional:**

- `LLM_MODEL_NAME`
- `VLLM_BASE_URL`
- `VLLM_API_KEY`
- `VERBOSE_DEBUG`

`packages.txt` installs Icarus Verilog at the system level automatically during Space build.

---

## Evaluation

`evaluate.py` runs a 28-prompt benchmark across three generation configurations. `make_charts.py` generates presentation figures from the results.

```bash
python evaluate.py
python make_charts.py
```

---

## Example Prompts

**Code generation:**

- Write a 4-bit synchronous counter with active-low reset.
- Design a 2-to-1 multiplexer using structural Verilog.
- Implement a full adder and verify it with a testbench.
- Create an 8-bit shift register with parallel load.

**Concept questions:**

- What is the difference between blocking and non-blocking assignments?
- Explain setup and hold time violations.
- How does a priority encoder work?

---

## License

This project is part of an M.S. Computer Science thesis at California State University, Fullerton. All rights reserved.
