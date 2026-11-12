"""Serve the EdgeCode browser interface and retrieval API."""
from pathlib import Path
import argparse
import json
import logging
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

class Engine:
    def __init__(self, model=None, lines=70, k=5, alpha=0.1):
        self.cfg = json.loads((ROOT/'config/pipeline.json').read_text())
        self.model = model or self.cfg['llm']['model']
        self.lines, self.k, self.alpha = lines, k, alpha
        self.status = 'starting'
        self.lock = threading.Lock()

    def initialize(self):
        try:
            from edgecode.generation import OllamaClient
            from edgecode.chunking import chunk_repository
            from edgecode.dense import DenseIndex
            from edgecode.bm25 import BM25Index
            self.client = OllamaClient(self.model, self.cfg['llm']['url'], timeout=600)
            self.client.check()
            repo = (ROOT/self.cfg['repository']['path']).resolve()
            if not repo.is_dir():
                raise ValueError('Missing Click source directory')
            self.status = 'indexing'
            chunks = chunk_repository(repo, strategy='ast', max_lines=self.lines,
                overlap_lines=self.cfg['overlap_lines'], include_prefixes=self.cfg['repository']['include_prefixes'])
            if not chunks:
                raise ValueError('No Python source found')
            self.dense = DenseIndex(chunks, model_name=self.cfg['embedding_model'], batch_size=self.cfg['embedding_batch_size'])
            self.lexical = BM25Index(chunks)
            self.status = 'ready'
        except Exception:
            logging.exception('EdgeCode startup failed. Check the corpus, Ollama model and dependencies.')
            self.status = 'unavailable'

    def ask(self, question):
        from edgecode.fusion import weighted_reciprocal_rank_fusion
        from edgecode.generation import answer_question
        start = time.monotonic()
        count = self.cfg['candidate_k']
        if self.alpha == 0:
            results = self.dense.search(question, k=count)[:self.k]
        elif self.alpha == 1:
            results = self.lexical.search(question, k=count)[:self.k]
        else:
            results = weighted_reciprocal_rank_fusion(
                {'dense': self.dense.search(question, k=count), 'bm25': self.lexical.search(question, k=count)},
                weights={'dense': 1-self.alpha, 'bm25': self.alpha}, rrf_k=self.cfg['rrf_k'], limit=self.k)
        answer = answer_question(self.client, question, results, max_chars=12000,
            num_ctx=self.cfg['llm']['num_ctx'], max_tokens=self.cfg['llm']['max_new_tokens'])
        chunks = {r.chunk_id: r.chunk for r in results}
        for source in answer['sources']:
            source['code'] = chunks[source['chunk_id']].content
        return dict(answer, seconds=round(time.monotonic()-start, 1), model=self.model)

def create_app(engine):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get('/')
    def index():
        return FileResponse(ROOT/'web/index.html')

    @app.get('/api/status')
    def status():
        return dict(status=engine.status, model=engine.model, corpus='Click 8.1.7')

    @app.post('/api/ask')
    def ask(payload: Question):
        question = payload.question.strip()
        if not question:
            raise HTTPException(422, 'Enter a question.')
        if engine.status != 'ready':
            raise HTTPException(503, 'The assistant is not ready. Please try again shortly.')
        if not engine.lock.acquire(blocking=False):
            raise HTTPException(429, 'The assistant is answering another question. Please try again shortly.')
        try:
            return engine.ask(question)
        except Exception:
            logging.exception('Answer generation failed')
            raise HTTPException(502, 'The answer could not be completed. Please try again.')
        finally:
            engine.lock.release()
    return app

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model')
    parser.add_argument('--lines', type=int, default=70)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    if args.lines <= 10 or not 1 <= args.k <= 20 or not 0 <= args.alpha <= 1:
        parser.error('Use lines > 10, K from 1 to 20, and alpha from 0 to 1.')
    import uvicorn
    engine = Engine(args.model, args.lines, args.k, args.alpha)
    threading.Thread(target=engine.initialize, daemon=True).start()
    uvicorn.run(create_app(engine), host=args.host, port=args.port)

if __name__ == '__main__':
    main()
