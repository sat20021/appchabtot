import sys
import io

# Fix Unicode encoding errors on Windows when printing emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Define paths to properly resolve imports from the inner directories
BASE_DIR = Path(__file__).resolve().parent
INNER_DIR = BASE_DIR / "Lead_agent_2.O" / "Lead_agent_2.O"
ML_MODELS_DIR = INNER_DIR / "Machine_Learning" / "ML_models"
FAQ_DIR = INNER_DIR / "faq_search_engine"
RAG_DIR = INNER_DIR / "RAG"

# Add directories to sys.path so modules can be imported
for directory in [INNER_DIR, ML_MODELS_DIR, FAQ_DIR, RAG_DIR]:
    if directory.exists() and str(directory) not in sys.path:
        sys.path.append(str(directory))

# Change current working directory to ML_models so relative model loading works
if ML_MODELS_DIR.exists():
    os.chdir(str(ML_MODELS_DIR))

# Import the unified server application which already has all the routes configured
try:
    from unified_server import app
    print("✅ Unified Server loaded successfully!")
    print("\n" + "="*50)
    print("🚀 MOUNTED API ROUTES:")
    print("="*50)
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        path = getattr(route, 'path', None)
        name = getattr(route, 'name', None)
        if methods and path:
            methods_str = ", ".join(methods)
            print(f"[{methods_str}] {path}  ({name})")
    print("="*50 + "\n")
except ImportError as e:
    print(f"❌ Failed to import unified_server: {e}")
    # Fallback to creating a new app if unified_server fails
    app = FastAPI(
        title="Lead Agent 2.0 API",
        description="Main entry point for all Lead Agent 2.0 routes",
        version="2.0.0"
    )
    
    @app.get("/")
    def fallback_root():
        return {"message": "Lead Agent 2.0 API (Fallback Mode)"}

if __name__ == "__main__":
    print("Starting Lead Agent 2.0 Main Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
