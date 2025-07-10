# 🧠 ADAMS RAG Evaluation System

A comprehensive RAG (Retrieval-Augmented Generation) system evaluation framework with AI agent-based dynamic metric selection.

## ✅ Features

- **📊 Data Annotation** - Data annotation and synthetic error generation
- **🤖 Agent-based Evaluation** - AI agents negotiate optimal evaluation metrics
- **⚖️ Real-time Weight Adjustment** - Dynamic metric weight adjustment
- **📚 History & Comparison** - Evaluation history and comparative analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API Key
- Hugging Face Token (read/write permissions)- for the backend script run.

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

### Startup Steps:

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

### 4. Access Application

- 🌐 **Frontend**: http://localhost:8501
- 🔗 **Backend API**: http://localhost:8000

> **💡 Startup Tips**:
>
> - Start backend first, then frontend
> - Keep both terminal windows running simultaneously
> - Backend shows: `Application startup complete.` when ready
> - Frontend shows: `You can now view your Streamlit app in your browser.`
> - Press `Ctrl+C` to stop each service

## 💡 Usage

### 📖 Comprehensive User Guide

For detailed step-by-step instructions, see our comprehensive guides:

- **📚 [User Guide](./USER_GUIDE.md)** - Detailed instructions, troubleshooting, and best practices.
- **📝 dataset to use for test: example_dataset**

### Basic Workflow

#### 1. Data Annotation Tab

- Upload CSV/JSON dataset (requires: `question`, `response`, `documents`, `key_points`)
- Run annotation pipeline to add synthetic errors and key points
- Download enhanced dataset for comprehensive evaluation

#### 2. Evaluation Tab

- Upload evaluation dataset
- Choose LLM model (`gpt-4o-mini` recommended)
- Start **Agent-based Evaluation** for AI-optimized metrics
- Adjust weights with real-time sliders
- View detailed results and AI discussion rationale

#### 3. History Tab

- Review past evaluations
- Compare different evaluation runs
