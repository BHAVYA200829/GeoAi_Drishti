#!/usr/bin/env bash
# Drishti quick start
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

echo "Starting Drishti on http://localhost:8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
