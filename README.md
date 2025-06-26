How to run the project:

poetry config virtualenvs.in-project true

poetry install

source .venv/bin/activate


poetry run python start_backend_poetry.py

poetry run streamlit run frontend/app.py --server.port 8501
