#!/usr/bin/env bash
set -e

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

streamlit run frontend/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true

wait $API_PID
