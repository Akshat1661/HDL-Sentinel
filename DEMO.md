# HDL Sentinel — Demo Guide

A quick walkthrough of what this app can do and what to try.

---

## What is HDL Sentinel?

HDL Sentinel is an AI assistant for Verilog hardware design. You describe what you want to build in plain English, and it:

- Writes the design module (the actual hardware)
- Writes a testbench (automated tests for the design)
- Compiles and runs the code automatically using Icarus Verilog
- If it fails, it reads the error and tries to fix it (up to 3 attempts)
- Shows you an interactive waveform of the simulation

No setup needed on your end — just type and go.

---

## The Two Main Modes

The app automatically figures out what you want based on your message.

| What you type                 | What happens                      |
| ----------------------------- | --------------------------------- |
| "Design a 4-bit counter"      | Generates + verifies Verilog code |
| "Explain what a flip-flop is" | Gives a theory explanation        |

---

## Try These Examples

### Code Generation

Type any of these into the chat box:

```
Design a 4-bit binary counter with synchronous reset and enable
```
```
Write a 2-to-1 multiplexer
```
```
Implement a D flip-flop with asynchronous reset
```
```
Design an 8-bit shift register with parallel load
```
```
Build a full adder
```
```
Design a 4-bit ALU that supports AND, OR, ADD, and SUB
```
```
Write a UART transmitter
```
```
Design a Moore FSM for a traffic light controller
```

What you will see:
1. A brief description of the design
2. The Verilog design module (copy-paste ready)
3. A testbench (shown in an expandable section)
4. A green checkmark if compilation and simulation passed, or an amber note if corrections were needed

---

### Concept / Theory Questions

Type any of these:

```
What is the difference between blocking and non-blocking assignments?
```
```
Explain setup time and hold time
```
```
What is metastability?
```
```
What is the difference between a latch and a flip-flop?
```
```
How does a synchronous reset differ from an asynchronous reset?
```
```
What is a finite state machine?
```
```
Explain clock domain crossing
```

What you will see: a plain-English explanation, sometimes with a short code snippet to illustrate the point.

---

### Off-Topic Rejection (this is intentional)

Try asking something unrelated to hardware:

```
What is the weather today?
```
```
Write me a Python script
```

The app will politely decline and stay focused on HDL topics.

---

## The Simulator Sandbox (Page 2)

This is your interactive workbench. Click the **Simulator Sandbox** tab at the top.

**What it does:** You get two side-by-side code editors — one for your design, one for your testbench. Hit **Run Simulation** and it compiles and runs right there.

**Try this:**
1. Go back to the chat and generate a counter design
2. Click the **"Open in Simulator"** button that appears below the code
3. The design and testbench get transferred automatically
4. Click **Run Simulation**
5. Scroll down to see the waveform chart — you can zoom in, hover over signals, and inspect every clock cycle
6. Click **Download VCD** if you want to open the waveform in GTKWave

**Manual editing:**
- Change a counter from 4-bit to 8-bit by editing the width in the code editor
- Add a new test case in the testbench
- Re-run to verify your changes

---

## Waveform Viewer

After running a simulation, an interactive waveform appears below the editors.

- **Hover** over any point to see signal values at that time
- **Zoom** using the range slider at the bottom
- Multi-bit signals (buses) show hex values annotated on the waveform
- Up to 24 signals are shown simultaneously

---

## Chat History (Requires Login)

If you create an account:

- Every conversation is saved automatically
- Previous chats appear in the left sidebar, newest first
- Click any chat to reload it
- Click the **x** button next to a chat to delete it
- Click **+ New Chat** to start fresh
- Your history persists across browser sessions and devices

---

## Good Demo Flow (5 minutes)

1. Type: `Design a 4-bit up-counter with synchronous reset`
   - Watch the code generate and verify in real time
2. Click **"Open in Simulator"** on the result
   - Hit Run Simulation, watch the waveform appear
3. Go back to chat, type: `What is the difference between blocking and non-blocking assignments?`
   - See the theory explanation (no code generated)
4. Try editing the counter width in the simulator from 4-bit to 8-bit and re-run

That covers code generation, automatic testing, waveform visualization, and concept explanation in about 5 minutes.

---

## Quick Reference — What the App Can Design

| Category      | Examples                                                     |
| ------------- | ------------------------------------------------------------ |
| Combinational | AND/OR/NAND gates, mux, decoder, encoder, adder, ALU         |
| Sequential    | D/T/JK flip-flop, registers, counters, LFSR, Johnson counter |
| Memory        | FIFO, RAM                                                    |
| FSMs          | Moore/Mealy state machines, traffic light, sequence detector |
| Communication | UART transmitter/receiver                                    |

---

## Things to Know

- The app only accepts Verilog 2001 — no SystemVerilog syntax
- Each prompt has a 1-second rate limit to prevent accidental spam
- The self-correction loop runs silently — if code fails on first try, it automatically retries up to 2 more times before reporting failure
- Simulation has a 30-second timeout (most designs finish in under 1 second)
