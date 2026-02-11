
# pg-agent

**pg-agent** is a signal-driven PostgreSQL incident intelligence system designed to help engineers detect early warning signs before they escalate into production outages.

Unlike traditional monitoring dashboards or generic AI assistants, pg-agent focuses on structured operational reasoning:

- Detect signals
- Validate evidence
- Perform root cause analysis
- Generate actionable runbooks only when confidence exists

---

## 🎯 Why pg-agent Exists

Most PostgreSQL incidents do not begin with outages.

They start with small operational signals:

- Long-running queries
- Blocking transactions
- Dead tuples and maintenance issues
- Misconfigured parameters
- Subtle performance degradation

Monitoring tools provide metrics — pg-agent focuses on **reasoning**.

---

## 🧠 Core Design Philosophy

pg-agent is intentionally designed around production reliability principles:

- Signal-first architecture (no assumptions without signals)
- Evidence-based decision making
- Explicit confidence levels
- Separation of detection, validation, and remediation
- Silence when insufficient data exists

The system is conservative by design to avoid hallucinated root causes.

---

## 🏗 Architecture Overview

User Input (CLI)
│
▼
Terminal Agent (Command Router)
│
▼
Signal Engine
│
▼
Evidence Builder
│
▼
Root Cause Engine
│
▼
Recommendation Engine
│
▼
Output Formatter / Reports
│
▼
Knowledge Base (Past Incidents)

---

## ⚙️ Key Components

| Component | Purpose |
|-----------|---------|
| TerminalAgent | Interactive CLI interface |
| SignalEngine | Detects operational risk signals |
| EvidenceBuilder | Validates signals with confidence scoring |
| RootCauseEngine | Identifies likely causes |
| Recommendation Engine | Generates runbooks and actions |
| Validator | LLM-assisted validation layer |
| ReportGenerator | HTML incident report generation |
| LlamaClient | Ollama-based LLM integration |

---

## ✨ Features

- CLI-first operational workflow
- Signal-based PostgreSQL analysis
- LLM-assisted reasoning via Ollama
- Knowledge-base driven recommendations
- Structured runbook generation
- HTML incident reporting
- Confidence scoring and validation

---

## 🚀 Quick Start

### Clone repository

```bash
git clone <repo-url>
cd pg-agent

Setup environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run interactive mode

python main.py


⸻

🧩 Example Commands

analyze SELECT * FROM orders;

search blocking transaction

runbook dead tuples cleanup

report postgres performance incident


⸻

🔄 Data Flow

CLI Input → Signal Detection → Evidence Validation → Root Cause Analysis → Recommendations → Report Output


⸻

📁 Project Structure

pg-agent/
├── main.py
├── agent/
├── core/
├── signals/
├── parser/
├── llm/
├── reports/
├── data/
├── tests/
└── README.md


⸻

🔮 Future Improvements
	•	EXPLAIN ANALYZE parsing support
	•	Integration with pg_stat_activity and pg_stat_statements
	•	Advanced confidence scoring
	•	Expanded incident knowledge base
	•	Automated remediation workflows

⸻

🧑‍💻 Development

Requirements:
	•	Python 3.8+
	•	Ollama (for LLM integration)
	•	macOS / Linux / Windows

⸻

📄 License

MIT License

---

# 🚀 Next Step (Strongly Recommended)

Now your repo looks professional — but we can make it **elite-level**.

If you want, next I can add:

🔥 Architecture diagram image embedded in README  
🔥 Senior-level badges (build, python, license, LLM)  
🔥 Demo GIF section (VERY powerful on LinkedIn)  
🔥 Engineering design philosophy section (this will impress CTOs heavily)

