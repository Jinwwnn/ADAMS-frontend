### ADAMS Interface

1. set up

poetry config virtualenvs.in-project true

poetry install

source .venv/bin/activate

#### `crete an .env` file：

```bash
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
ANSWER_TYPE=response
```



2. run the project

cd backend
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

cd frontend
streamlit run app.py --server.port 8502
