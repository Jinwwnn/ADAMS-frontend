# 🧠 ADAMS RAG Evaluation System

A comprehensive RAG (Retrieval-Augmented Generation) system evaluation framework with AI agent-based dynamic metric selection.

## ✅ Features

- **📊 Data Annotation** - Data annotation and synthetic error generation
- **🤖 Agent-based Evaluation** - AI agents negotiate optimal evaluation metrics
- **📈 Standard Evaluation** - Traditional metric-based evaluation
- **📚 History & Comparison** - Evaluation history and comparative analysis
- **⚖️ Real-time Weight Adjustment** - Dynamic metric weight adjustment

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API Key
- Hugging Face Token (read/write permissions)

### 1. Environment Setup

Create `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
ANSWER_TYPE=gold
```

### 2. Install Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install streamlit fastapi uvicorn pandas requests
```

### 3. Start System

#### Option A: Automatic Startup (One-click)

```bash
python start_services.py
```

#### Option B: Manual Startup (Recommended, Faster)

**Backend**:

```bash
cd backend
python -m uvicorn api.app:app --host localhost --port 8000 --reload
```

 **Frontend** - new terminal:

```bash
cd frontend  
streamlit run app.py --server.port 8501 --server.address localhost
```

**Verify Services are Running**:
```bash
# Quick status check (recommended)
python start_services.py status

# Manual check
curl http://localhost:8000  # Backend
curl http://localhost:8501  # Frontend

# Or visit in browser
open http://localhost:8501
```

### 4. Access Application

- 🌐 **Frontend**: http://localhost:8501
- 🔗 **Backend API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

> **💡 Startup Tips**:
>
> - Manual startup is faster and allows real-time log viewing
> - Start backend first, then frontend
> - Keep both terminal windows running simultaneously  
> - Backend shows: `Application startup complete.` when ready
> - Frontend shows: `You can now view your Streamlit app in your browser.`
> - Test connection: Visit http://localhost:8501 and click "Test OpenAI API"
> - Check status anytime: `python start_services.py status`
> - Press `Ctrl+C` to stop each service

## 💡 Usage

### 1. Data Annotation Tab

- Upload CSV dataset (requires: `question`, `response`, `documents`)
- Select error types for synthetic error generation
- Configure error probabilities
- Generate enhanced dataset with key points and error answers

### 2. Evaluation Tab

**Standard Evaluation**: Use predefined metrics

**Agent-based Dynamic Evaluation**:

- Describe your evaluation requirements
- AI agents discuss and select optimal metrics
- View discussion results and selected weights
- Real-time weight adjustment

### 3. History Tab

- View all evaluation records
- Compare two evaluations in detail
- Export comparison reports

## 📊 Supported Metrics

- Answer Equivalence
- Factual Correctness
- BERTScore
- Learning Facilitation
- Engagement
- Context Relevance
- Key Point Analysis (Completeness/Irrelevance/Hallucination)
- Adherence Faithfulness
- Context Utilization
- Coherence
- Factual Accuracy
- Refusal Accuracy

## 🔧 Technical Architecture

- **Backend**: FastAPI with 13+ evaluation metrics
- **Frontend**: Streamlit with cyberpunk-style UI
- **Agent System**: AutoGen-based multi-agent negotiation
- **Storage**: Local JSON-based evaluation history

## ⚠️ Notes

- Agent evaluation requires more OpenAI API tokens
- First agent evaluation takes 1-3 minutes for discussion
- Evaluation history stored locally in `backend/evaluation_history.json`

## 📞 Support

For issues, please provide:

1. Error screenshots
2. Dataset format
3. Selected evaluation mode
4. System logs

---

**License**: MIT
**Version**: 3.0.0

Enjoy using ADAMS! 🎉
