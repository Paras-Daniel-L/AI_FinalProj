"""
Entrypoint for the Sagot AI RAG chatbot API.

Run from the project root with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000

(This file must stay at the project root — it's what turns `app/` from
a loose folder of scripts into something runnable with a single command.)
"""

from app.api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
