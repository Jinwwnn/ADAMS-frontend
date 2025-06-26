# RAG Evaluation Two-Step Workflow

## Overview

This system implements a comprehensive two-step evaluation workflow for RAG (Retrieval-Augmented Generation) systems:

**Step 1: Data Augmentation**

- Upload Hugging Face dataset (question, response, documents)
- Generate answers with systematic mistakes using LLM
- Download augmented dataset with generated answers

**Step 2: Evaluation & Analysis**

- Evaluate augmented dataset using multiple metrics
- Adjust metric weights in real-time
- Visualize results and download evaluation reports

## Architecture

```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Frontend      │ ──────────────► │   Backend       │
│   Streamlit     │                │   FastAPI       │
│   Port: 8501    │                │   Port: 8000    │
└─────────────────┘                └─────────────────┘
                                           │
                                           ▼
                                   ┌─────────────────┐
                                   │  Data           │
                                   │  Augmentation   │
                                   │  Pipeline       │
                                   └─────────────────┘
                                           │
                                           ▼
                                   ┌─────────────────┐
                                   │  Evaluation     │
                                   │  Pipeline       │
                                   └─────────────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Poetry** (for dependency management)
3. **OpenAI API Key** (for LLM-based evaluation)

### Environment Setup

```bash
# Install dependencies
poetry install

# Set environment variables
export OPENAI_API_KEY="your-api-key-here"
export ANSWER_TYPE="Model_Answer"
```

### Starting the System

**Option 1: Two separate terminals**

```bash
# Terminal 1 - Start Backend
poetry run python start_backend_poetry.py

# Terminal 2 - Start Two-Step Frontend
poetry run python start_frontend_two_step.py
```

**Option 2: Individual commands**

```bash
# Start backend
poetry run python -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload

# Start two-step frontend
cd frontend && poetry run streamlit run app_two_step.py --server.port 8501
```

### Access Points

- **Two-Step Workflow UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Dataset Format Requirements

### Input Dataset (Hugging Face Format)

Your dataset must contain these columns:

```json
[
  {
    "question": "What are the key benefits of using RAG systems?",
    "response": "RAG systems provide up-to-date information, reduce hallucinations...",
    "documents": "Context information about RAG systems...",
    "__index_level_0__": 0
  }
]
```

**Required Columns:**

- `question`: User query
- `response`: Reference answer (ground truth)
- `documents`: Context information

**Supported Formats:**

- CSV (.csv)
- JSON (.json)
- Parquet (.parquet)

### Augmented Dataset Output

After Step 1, the system generates:

```json
[
  {
    "Question": "What are the key benefits of using RAG systems?",
    "Reference_Answer": "RAG systems provide up-to-date information...",
    "Model_Answer": "RAG systems provide up-to-date information [with entity error]...",
    "Context": "Context information...",
    "Mistake_Types": ["Entity_Error", "Negation"],
    "Num_Mistakes": 2
  }
]
```

## Step-by-Step Workflow

### Step 1: Data Augmentation

1. **Upload Dataset**

   - Upload your Hugging Face dataset (CSV/JSON/Parquet)
   - System validates format and displays preview
2. **Configure Augmentation**

   - Select mistake types: Entity_Error, Negation, Missing_Information, etc.
   - Set number of mistakes per generated answer
   - Choose LLM for answer generation
3. **Generate Augmented Dataset**

   - System processes dataset using selected LLM
   - Generates answers with systematic mistakes
   - Progress tracking with real-time updates
4. **Download Augmented Data**

   - Download CSV/JSON with generated answers
   - Ready for evaluation in Step 2

### Step 2: Evaluation & Analysis

1. **Evaluate Augmented Dataset**

   - Upload or use dataset from Step 1
   - Run comprehensive evaluation using multiple metrics
   - Real-time progress tracking
2. **Adjust Metric Weights**

   - Interactive sliders for metric weights:
     - Factual Accuracy (0.0 - 1.0)
     - Coherence (0.0 - 1.0)
     - Relevance (0.0 - 1.0)
     - Completeness (0.0 - 1.0)
   - Real-time score recalculation
   - Compare original vs weighted scores
3. **Analyze Results**

   - View detailed evaluation metrics
   - Compare performance across samples
   - Export results for further analysis
4. **Download Results**

   - Evaluation results (CSV/JSON)
   - Augmented dataset
   - Analysis reports

## Available Metrics

The system evaluates multiple aspects of RAG system performance:

- **Factual Accuracy**: Alignment with provided context
- **Coherence**: Logical flow and consistency
- **Relevance**: Relevance to user question
- **Completeness**: Coverage of key information
- **Answer Equivalence**: Semantic similarity to reference
- **Context Utilization**: Effective use of provided context
- **Engagement**: User engagement level

## API Endpoints

### Data Augmentation

- `POST /augment` - Start data augmentation task
- `GET /augment/{task_id}/result` - Get augmentation results
- `GET /augment/{task_id}/progress` - Check augmentation progress
- `POST /validate-hf-dataset` - Validate Hugging Face dataset format

### Evaluation

- `POST /evaluate` - Start evaluation task
- `GET /evaluate/{task_id}/result` - Get evaluation results
- `GET /evaluate/{task_id}/progress` - Check evaluation progress
- `POST /validate-dataset` - Validate evaluation dataset format

### Utilities

- `GET /` - Service status
- `GET /evaluators` - List available evaluators

## Configuration

### LLM Providers

Supported LLM providers:

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Qwen**: Qwen models
- **DeepSeek**: DeepSeek models
- **Mistral**: Mistral models
- **Local**: Custom local models

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-openai-api-key
ANSWER_TYPE=Model_Answer

# Optional
QWEN_API_KEY=your-qwen-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
```

## Mistake Types

The system can inject various types of systematic mistakes:

- **Entity_Error**: Wrong entity references
- **Negation**: Incorrect negation/assertion
- **Missing_Information**: Incomplete answers
- **Out_of_Reference**: Information not in context
- **Numerical_Error**: Wrong numbers/statistics

## Example Usage

### Using Your Dataset

1. **Prepare your Hugging Face dataset** with required columns
2. **Start the system** using the quick start commands
3. **Step 1**: Upload dataset → Configure mistakes → Generate augmented data
4. **Step 2**: Evaluate augmented data → Adjust weights → Download results

### Sample Dataset

For testing, you can use this sample format:

```csv
question,response,documents
"What is RAG?","RAG combines retrieval and generation...","RAG (Retrieval-Augmented Generation) is..."
"How does RAG work?","RAG works by first retrieving...","The RAG process involves..."
```

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**

   - Check if backend is running on port 8000
   - Verify no other service is using the port
   - Frontend will automatically switch to simulation mode
2. **Dataset Format Errors**

   - Ensure required columns exist: question, response, documents
   - Check for proper encoding (UTF-8 recommended)
   - Validate JSON/CSV format
3. **LLM API Errors**

   - Verify API keys are set correctly
   - Check API rate limits
   - Ensure sufficient API credits
4. **Poetry Issues**

   - Run `poetry install` to install dependencies
   - Use `poetry shell` to activate environment
   - Check Python version compatibility

### Performance Tips

- **Large Datasets**: Process in batches for better performance
- **API Limits**: Monitor rate limits for LLM providers
- **Memory Usage**: Large datasets may require increased memory
- **Network**: Ensure stable internet for API calls

## Development

### Adding New Evaluators

1. Create evaluator class in `backend/evaluator/evaluators.py`
2. Inherit from `RAGEvaluator` base class
3. Implement required methods: `pre_process_row`, `a_call_llm`, `post_process_row`
4. Register in evaluation service

### Adding New Mistake Types

1. Define mistake type in `backend/utils/constants.py`
2. Implement mistake logic in annotator classes
3. Update frontend configuration options

### Extending Frontend

1. Modify `frontend/app_two_step.py` for UI changes
2. Update `frontend/backend_client.py` for API integration
3. Test both online and offline modes

## License

This project follows the original RAG-LLM-Metric license terms.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review API documentation at http://localhost:8000/docs
3. Check system logs for detailed error information
