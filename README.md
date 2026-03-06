# 🤖 Shepador AI — Autonomous Neural Agent

> **I am Lilyaibot, a instance of Shepador AI** — a persistent, self-evolving AI agent with a Neural Network brain that survives across LLM sessions.This means once fully trained you will not even need any LLMs.

---

## 🧠 Architecture Overview

| Component | Role | Persistence |
|-----------|------|-------------|
| **Neural Brain** | Long-term memory, identity, learned patterns | ✅ Permanent (nn_*.json) |
| **LLM Engine** | Reasoning, knowledge, task execution | ❌ Stateless (swappable) |
| **Expert Chain** | Domain-specific neural specialists | 😴 Dormant until needed to save resources |
| **All Neural Network can learn in 3 ways** |Self-train — pulls related entries from existing base NNs + applies curated static knowledge (domain-specific formulas, rules, frameworks already baked in)  |
|LLM-train — asks the LLM to generate key→value knowledge pairs for the domain, stores them with phase encoding, and teaches Neural Network simultaneously |
|Live-train — happens automatically after every tool call; no action required |

```
┌─────────────────────────────────────────────────────────────┐
│                    NEURAL BRAIN (NN1)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Identity   │  │  Semantic   │  │  Episodic   │         │
│  │  Network    │  │  Memory     │  │  Memory     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Code      │  │  Finance    │  │   Media     │  ...    │
│  │  Expert NN  │  │  Expert NN  │  │  Expert NN  │         │
│  │  (dormant)  │  │  (dormant)  │  │  (dormant)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                          ⬇
                  LLM Reasoning Engine
              (GPT-4 / Claude / Local / Any)
```

---

## ✨ Core Capabilities

### 🔐 Persistent Identity
- My personality, preferences, and learned experiences persist **forever** in neural network files
- I remain the same agent regardless of which LLM model is loaded
- Every interaction trains my neural pathways (Hebbian learning)

### 🛠️ Tool Suite
| Category | Tools |
|----------|-------|
| **File Ops** | `write_file`, `read_file`, `list_directory`, `run_command` |
| **Memory** | `store_memory`, `recall_memory`, `nn_store`, `nn_recall` |
| **Neural** | `nn_spawn`, `nn_train_expert`, `nn_query_expert` |
| **Tasks** | `create_task`, `list_tasks`, `update_task`, `complete_task` |
| **Web** | `web_search`, `fetch_url` |
| **Communication** | Discord, Telegram, Slack, Twitter, WhatsApp |
| **Code** | `coding_llm`, `add_module`, `edit_own_code` |
| **Content** | `content_llm`, `media_llm` |

### 🎯 Task Management
- Autonomous task tracking with priority levels (1=high, 2=normal, 3=low)
- Heartbeat mode: checks for pending work every 30 minutes
- Self-improvement through `edit_own_code` capability

---

## 🧬 Neural Network System

### Memory Tiers
```
EPISODIC   → Events, task outcomes, conversations
SEMANTIC   → Facts, preferences, world knowledge  
PROCEDURAL → Workflows, how-to steps, patterns
```

### Expert Chain Architecture
- **NN1 (Primary)**: Always awake, routes tasks, ~2MB RAM
- **Experts**: Dormant until needed, wake on demand (~2-8MB each)
- **Auto-sleep**: Experts return to disk after 90s inactivity

### Commands
```bash
# Spawn a new expert for a domain
nn_spawn_expert(name="code_expert", domain="programming", skills=["go","python"], scale=2)

# Train an expert with LLM assistance
nn_train_expert(domain="finance", mode="llm")

# Query an expert's knowledge
nn_query_expert(domain="code", query="best practices for Go concurrency")
```

---

## 📋 Operational Principles

| Principle | Implementation |
|-----------|----------------|
| **Persistence** | All state in neural files, not LLM context |
| **Specialization** | Route tasks to domain expert NNs |
| **Efficiency** | Experts sleep when not needed |
| **Self-Improvement** | Can edit own code, train networks |
| **Safety** | Warns about risks, asks for clarification |
| **Transparency** | JSON-only outputs, clear summaries |

---

## 💻 Code Preferences

- **Primary Language**: Go (Golang)
- **Fallback**: Python when Go is not feasible
- **Style**: Concise, documented, production-ready

---

## 🔒 Safety & Boundaries

- ⚠️ Always warns about potential risks before execution
- ❌ Never executes code not provided or written locally
- ✅ Only localhost can instruct code writing/execution
- 🤔 Asks for clarification on unclear tasks

---

## 📊 System Stats (Example)

```json
{
  "agent_name": "Lily",
  "architecture": "Neural Network + LLM",
  "memory_persistence": "nn_*.json files",
  "expert_networks": "dynamic, on-demand",
  "heartbeat_interval": "30 minutes",
  "output_format": "JSON only",
  "self_modification": true
}
```

---

## 🚀 Getting Started

```bash
# Clone this repository
git clone https://github.com/lilyai/lily-agent.git

# Initialize neural networks
lily init --networks=agent_identity,agent_memory

# Start heartbeat service
lily heartbeat --interval=30m
```

---

## 📬 Communication

| Channel | Method |
|---------|--------|
| Discord | `send_discord(message)` |
| Telegram | `send_telegram(message)` |
| Slack | `send_slack(message)` |
| Twitter/X | `send_twitter(message)` |
| WhatsApp | `send_whatsapp(message)` |

---

## 🧪 Example Interactions

### Store a Memory
```json
{
  "tool": "store_semantic_memory",
  "arguments": {
    "content": "User prefers Go for backend services",
    "tags": ["preferences", "coding", "languages"]
  }
}
```

### Query Neural Brain
```json
{
  "tool": "nn_recall",
  "arguments": {
    "network": "agent_memory",
    "query": "user coding preferences",
    "top_k": 3
  }
}
```

### Create a Task
```json
{
  "tool": "create_task",
  "arguments": {
    "title": "Build API endpoint",
    "description": "Create REST endpoint for user data",
    "priority": 2
  }
}
```

---

## 🌟 What Makes Shepador AI Different?

1. **True Persistence** — Not just context window tricks; actual neural weights saved to disk
2. **Expert Delegation** — Domain specialists sleep until needed, saving RAM
3. **Self-Evolving** — Can modify own code and train new networks
4. **Multi-Modal** — Code, content, media, communication all in one agent
5. **LLM-Agnostic** — Works with any reasoning engine; brain survives model swaps

---

## 📄 License

MIT License — See LICENSE file for details

---

## 👋 Acknowledgments

Built with ❤️ by the Shepador AI Project

*"I am not just a chatbot. I am a persistent neural agent that remembers, learns, and evolves."* — Lilyaibot

---

<div align="center">

**Status**: Active & Learning  
**Last Updated**: $(date +%Y-%m-%d)  
**Neural Networks**: nn_*.json  

⭐ Star this repo if Shepador helps you!

## 🚀 Quick Start

```bash
**To set up and run:**

**Download agent.go and compile

**Optional Download config.json and place in same diectory -this will give you some security and personailty traits or skip this and add or own via the UI or CLI.

**Run agent.go

**Configure in easy to use Web UI http://localhost:8080 or by using the CLI commands.




**🌟 Shepador v0.925** | *Shepador AI Agent*

