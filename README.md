# pg-agent

A Python-based agent system that integrates with LLM (Llama), handles signal processing, parses inputs, validates agent outputs, and generates reports.

## 📋 Overview

pg-agent is an intelligent agent system that combines:
- **CLI Interface** for interactive command processing
- **LLM Integration** powered by Llama models via Ollama API
- **Signal Processing** pipeline for data analysis
- **Output Validation** using AI reasoning
- **Report Generation** with HTML formatted incident reports

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd pg-agent
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Run the Project
```bash
python3 main.py
```

---

## 📁 Project Structure

```
pg-agent/
├── main.py                      # Entry point with CLI argument parsing
├── README.md                    # This file
├── SETUP.md                     # Detailed setup guide
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
│
├── agent/                       # Agent modules
│   ├── __init__.py
│   ├── terminal_agent.py       # CLI loop for user interaction
│   └── validator.py            # Validates agent outputs using LLM
│
├── collectors/                  # Data collectors
│   └── __init__.py
│
├── data/                        # Data directories
│   ├── raw/                     # Raw input data
│   ├── normalized/              # Normalized data
│   └── output/                  # Output files (reports, results)
│
├── llm/                         # LLM integration
│   ├── __init__.py
│   └── llama_client.py         # Ollama API client for Llama models
│
├── parser/                      # Input parsing
│   ├── __init__.py
│   └── input_parser.py         # CLI argument parsing
│
├── reports/                     # Report generation
│   ├── __init__.py
│   ├── report_generator.py     # HTML report generation
│   └── templates/
│       └── incident_report.html # Jinja2 report template
│
└── signals/                     # Signal processing
    ├── __init__.py
    ├── signal_engine.py        # Signal pipeline (generate → analyze → filter)
    └── evidence_builder.py     # Evidence collection with confidence scores
```

---

## 🧩 Key Modules

| Module | File | Purpose |
|--------|------|---------|
| **InputParser** | `parser/input_parser.py` | Parses CLI arguments: `--input`, `--output`, `--log-level` |
| **LlamaClient** | `llm/llama_client.py` | Interacts with Ollama API (`http://localhost:11434`) |
| **SignalEngine** | `signals/signal_engine.py` | Processes data through signal pipeline (generate → analyze → filter) |
| **EvidenceBuilder** | `signals/evidence_builder.py` | Collects evidence from signal results with confidence scores |
| **TerminalAgent** | `agent/terminal_agent.py` | Main CLI loop with command processing |
| **Validator** | `agent/validator.py` | Validates agent responses using LLM reasoning |
| **ReportGenerator** | `reports/report_generator.py` | Generates HTML incident reports with Jinja2 templates |

---

## 🔄 Data Flow

```
CLI Input → InputParser → SignalEngine → EvidenceBuilder → TerminalAgent → Validator → ReportGenerator → HTML Report
                        ↓
                  LlamaClient (ollama)
```

**Flow Steps:**
1. **InputParser** - Parses user arguments and input files
2. **SignalEngine** - Generates signals, analyzes data, filters results
3. **EvidenceBuilder** - Collects evidence with confidence scores
4. **TerminalAgent** - Processes commands and interactions
5. **Validator** - Validates outputs using LLM reasoning
6. **ReportGenerator** - Creates formatted HTML reports

---

## ✨ Features

- 🤖 **Terminal Agent** - Interactive command-line interface for agent interactions
- 🧠 **LLM Integration** - Llama model client for AI-powered processing via Ollama
- 📊 **Signal Engine** - Process and analyze signals with pipeline stages
- 📝 **Input Parser** - Parse and normalize CLI input data
- ✅ **Validator** - Validate agent outputs using LLM reasoning
- 📑 **Report Generator** - Generate detailed HTML incident reports
- 🎨 **Jinja2 Templates** - Dynamic report generation with HTML templates
- 📁 **Data Management** - Raw, normalized, and output data handling

---

## 🛠️ Requirements

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python Version**: 3.8+ (3.10+ recommended)
- **RAM**: Minimum 4GB (8GB+ recommended for LLM operations)
- **Disk Space**: At least 1GB free space

### Required Software
- Python 3.8+
- pip3
- Git
- **Ollama** (for LLM integration) - Download from https://ollama.com

### Python Dependencies
```
# Core Dependencies
requests>=2.28.0
python-dotenv>=1.0.0
pyyaml>=6.0
colorlog>=6.7.0

# LLM Dependencies
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Web Framework
flask>=2.3.0
jinja2>=3.1.0

# Validation & Testing
pydantic>=2.0.0
pytest>=7.0.0
pytest-cov>=4.0.0

# Development
black>=23.0.0
mypy>=1.0.0
pre-commit>=3.0.0
watchfiles>=0.19.0
```

---

## 📖 Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup instructions, configuration options, and development guide
- **Module Docstrings** - Each Python file contains detailed docstrings

---

## 💻 Usage Examples

### Basic Usage
```bash
python3 main.py --input data/raw/input.txt --output data/output/
```

### With Custom Log Level
```bash
python3 main.py --input data/raw/data.json --output data/output/ --log-level DEBUG
```

### Run in Interactive Mode
```bash
python3 main.py
# Then enter commands in the interactive terminal
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests with `pytest`
5. Submit a pull request

---

## ✅ Quality & Testing Status

### Test Coverage
- **Total Tests**: 130
- **Pass Rate**: 100% ✓
- **Modules Tested**:
  - Core signal detection and root cause analysis
  - Confidence scoring and evidence building
  - Clarification engine for low-confidence scenarios
  - Recommendation review & approval workflows
  - Terminal agent and command processing

### Code Quality
- **Python Files**: 50 (all syntax-valid)
- **Zero Duplicate Code**: Single source of truth for all models
- **Package Exports**: Unified and consistent
- **Dependencies**: Complete and documented in requirements.txt

### Recent Cleanup (Jan 28, 2026)
- ✓ Removed 11 duplicate documentation files
- ✓ Removed 1 empty directory (collectors/)
- ✓ Removed 2 orphaned files
- ✓ Created missing review_renderer.py module
- ✓ Updated package exports for review/approval system
- ✓ All 130 tests passing

See [PROJECT_CLEANUP_SUMMARY.md](PROJECT_CLEANUP_SUMMARY.md) for detailed cleanup report.

---

**pg-agent** - An intelligent agent system for signal processing and AI-powered analysis.

