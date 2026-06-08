FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir . || pip install --no-cache-dir numpy torch --extra-index-url https://download.pytorch.org/whl/cpu fastapi uvicorn websockets plotly pandas pydantic aiofiles streamlit matplotlib

COPY . .

RUN mkdir -p data/models data/games

EXPOSE 8000 8501

ENV MCTS_CHESS_DATA_DIR=/app/data

CMD uvicorn mcts_chess.api.app:app --host 0.0.0.0 --port 8000 & streamlit run mcts_chess/visualization/dashboard.py --server.port 8501 --server.address 0.0.0.0
