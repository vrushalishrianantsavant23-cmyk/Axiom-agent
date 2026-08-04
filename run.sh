
#!/bin/bash
cd /workspaces/Axiom-agent
source .venv/bin/activate

fuser -k 8000/tcp 2>/dev/null
fuser -k 8501/tcp 2>/dev/null
sleep 2

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
streamlit run streamlit_app.py --server.port 8501
