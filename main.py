"""
Entrypoint for the Sagot AI RAG chatbot API.

Run from the project root with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000          (the chatbot)
      and http://localhost:8000/eval     (the evaluation tool)
      and http://localhost:8000/demo     (the live demo for the panel)

(This file must stay at the project root — it's what turns `app/` from
a loose folder of scripts into something runnable with a single command.)
"""

from app.api import app

# The evaluation tool (rag_eval/web.py) is added here rather than inside
# app/api.py, so the chatbot never depends on the evaluation code: if the
# evaluation extras (scipy, numpy — rag_eval/requirements.txt) aren't
# installed, the chatbot still starts and only /eval is missing.
try:
    from rag_eval.web import router as eval_router
    app.include_router(eval_router)
    # The live demo page (/demo): chat + live process view + three metrics,
    # with the C0/C1/C2 comparisons (rag_eval/demo_web.py, app/baselines.py).
    from rag_eval.demo_web import router as demo_router
    app.include_router(demo_router)
except ImportError as e:
    print(f"⚠️  [Eval] evaluation tool not loaded ({e}). "
          "Install it with: pip install -r rag_eval/requirements.txt")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
