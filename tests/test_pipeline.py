import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from edgecode.workflow import Workflow, read, write, run_llm, citations, load_questions, generate
from edgecode.generation import inspect_response
from edgecode.reporting import select_k, budget_context, context_tokens
from edgecode.models import Chunk, SearchResult

ROOT=Path(__file__).resolve().parents[1]

class FakeEmbedder:
    def embed(self,texts,**kwargs):
        return np.array([[t.lower().count('alpha'),t.lower().count('beta'),1] for t in texts],dtype=np.float32)
    query_embed=embed

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'repo/pkg').mkdir(parents=True)
        (self.root/'repo/pkg/sample.py').write_text('def alpha():\n    return 1\n\n'+'\n'.join('x = '+str(i) for i in range(160))+'\ndef beta():\n    return 2\n')
        cats=['exact_identifier','semantic_behaviour','cross_function_or_file','error_or_edge_case','code_explanation']
        def questions(n,split):
            return [dict(id=f'{split}_{i}',question=f'{split} alpha question {i}',category=cats[i%5],split=split,
                gold_spans=[dict(path='pkg/sample.py',start_line=1,end_line=2)]) for i in range(n)]
        write(self.root/'data/dev.json',questions(20,'dev'));write(self.root/'data/eval.json',questions(5,'test'))
        cfg=read(ROOT/'config/pipeline.json');cfg.update(repository=dict(path='repo',include_prefixes=['pkg/'],id='fixture',version='1'),
            development_questions='data/dev.json',evaluation_questions='data/eval.json',expected_evaluation_count=5)
        cfg['timing']=dict(warmups=0,repeats=1)
        write(self.root/'config/pipeline.json',cfg)
        self.flow=Workflow(self.root/'config/pipeline.json',embedder=FakeEmbedder(),verbose=False)
    def tearDown(self):self.tmp.cleanup()
    def test_complete_smoke_and_frozen_evaluation(self):
        with self.assertRaises(ValueError):self.flow.evaluate()
        # Step 1 must not invoke either BM25 search or fusion.
        with patch('edgecode.workflow.BM25Index',side_effect=AssertionError('No BM25 in sizes')):
            self.flow.sizes()
        s=self.flow.state();folder=Path(s['stage_outputs']['sizes'])
        with (folder/'summary.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),8);self.assertTrue((folder/'mrr_by_size.png').is_file())
        self.flow.alpha();self.flow.k()
        self.assertEqual(self.flow.state()['status'], 'k_pending')
        with self.assertRaisesRegex(ValueError, 'selected manually'):
            self.flow.evaluate()
        out=self.flow.evaluate(selected_k=5)
        frozen=read(self.flow.statefile)
        self.assertEqual(frozen['k'], 5)
        self.assertEqual(frozen['status'], 'ready')
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            self.flow.finalize_k(frozen, 3)
        self.flow.evaluate()
        self.assertEqual(frozen,read(self.flow.statefile))
        self.assertEqual(len(read(out/'per_question.json')),15)
        self.assertEqual(set(r['method'] for r in read(out/'per_question.json')),{'fixed_dense','ast_dense','ast_hybrid'})
        self.assertTrue((out/'categories_recall_at_k.png').is_file())
        args=argparse.Namespace(evaluation_dir=str(out),limit=5,fail_fast=True)
        def fake_generate(*args):
            return dict(answer='Returns one [S1].',generation_status='completed',generation_seconds=.001,
                        prompt_eval_count=12,eval_count=5,raw_response={'done':True})
        with patch('edgecode.workflow.model_metadata',return_value={'test':True}),patch('edgecode.workflow.generate',side_effect=fake_generate):
            llm=run_llm(self.flow,args)
        generated=read(llm/'results.json');self.assertEqual(len(generated),15)
        self.assertTrue(all(r['citation_ids_valid']==1 for r in generated))
        self.assertTrue(all(r['context_tokens']<=3000 for r in generated))
        self.assertTrue((llm/'fixed_dense.json').exists())
        with (llm/'manual_review.csv').open() as f:
            self.assertTrue(all(r['correctness_0_2']=='' for r in csv.DictReader(f)))
        self.flow.sizes()
        with self.assertRaises(ValueError):self.flow.state(ready=True)
    def test_development_does_not_read_evaluation(self):
        (self.root/'data/eval.json').write_text('malformed evaluation')
        self.flow.sizes();self.flow.alpha();self.flow.k()
        self.assertEqual(self.flow.state()['status'],'k_pending')
    def test_leakage_and_malformed_gold(self):
        q=dict(self.flow.dev[0],split='test')
        write(self.root/'leak.json',[q])
        with self.assertRaisesRegex(ValueError,'overlaps'):load_questions(self.root/'leak.json',self.flow.repo,self.flow.prefixes,dev_questions=self.flow.dev)
        q.update(id='fresh',question='new unrelated',gold_spans=[dict(path='pkg/sample.py',start_line=1,end_line=9999)])
        write(self.root/'leak.json',[q])
        with self.assertRaisesRegex(ValueError,'past end'):load_questions(self.root/'leak.json',self.flow.repo,self.flow.prefixes)
    def test_source_change_rejected(self):
        self.flow.sizes();(self.root/'repo/pkg/sample.py').write_text('x=1\n')
        with self.assertRaises(ValueError):Workflow(self.root/'config/pipeline.json',embedder=FakeEmbedder()).state()
    def test_endpoints(self):
        q=self.flow.dev[0]['question'];dense,_=self.flow.index('ast',50)
        self.assertEqual([r.chunk_id for r in self.flow.retrieve(q,'ast',50,0)], [r.chunk_id for r in dense.search(q,k=20)])
        lexical=self.flow.lexical('ast',50)
        self.assertEqual([r.chunk_id for r in self.flow.retrieve(q,'ast',50,1)], [r.chunk_id for r in lexical.search(q,k=20)])
    def test_response_and_citation_classification(self):
        for content,reason,status in [('', 'stop','empty_response'),('partial','length','output_token_limit'),('answer','stop','completed')]:
            self.assertEqual(inspect_response(dict(message={'content':content},done=True,done_reason=reason))['generation_status'],status)
        sources=[dict(id='S1',start_line=10,end_line=20)]
        self.assertEqual(citations('[S1:L10-L20]',sources)['citation_ids_valid'],1)
        for text in ('[S2]','[S1:1-99]','[S1:garbage]','No references'):
            self.assertEqual(citations(text,sources)['citation_ids_valid'],0)
    def test_k_tolerance_prefers_small(self):
        rows=[dict(method=m,k=k,recall_at_k=r,precision_at_k=.3,context_tokens=100*k,retrieval_seconds=.01)
              for m in ('fixed_dense','ast_dense','ast_hybrid') for k,r in ((3,.89),(5,.90),(10,.90))]
        k,_=select_k(rows,self.flow.cfg['k_selection']);self.assertEqual(k,3)
        rules=dict(self.flow.cfg['k_selection'],max_context_tokens=1)
        with self.assertRaises(ValueError):select_k(rows,rules)

if __name__=='__main__':unittest.main()

class CliAndTransportTests(unittest.TestCase):
    def test_original_ask_check(self):
        import ask
        with patch('sys.argv',['ask.py','--model','fixture','--check']),patch('ask.OllamaClient.check',return_value='test'):
            self.assertEqual(ask.main(),0)
    def test_generation_payload_and_budget(self):
        from edgecode.generation import OllamaClient
        options=dict(read(ROOT/'config/pipeline.json')['llm'],seed=42)
        client=OllamaClient('fixture')
        with patch.object(client,'request',return_value=dict(done=True,message={'content':'answer [S1]'},eval_count=5)) as req:
            r=generate(client,'question','[S1] source',options)
        self.assertEqual(r['generation_status'],'completed')
        payload=req.call_args.args[1]
        self.assertFalse(payload['think']);self.assertEqual(payload['options']['num_predict'],1024)
        self.assertEqual(payload['options']['temperature'],0)
        with patch.object(client,'request') as req:
            r=generate(client,'question','x'*20000,options)
            req.assert_not_called();self.assertEqual(r['generation_status'],'prompt_budget_exceeded')
    def test_llm_preflight_mock(self):
        from edgecode.workflow import llm_check
        with TemporaryDirectory() as d:
            cfg=Path(d)/'config/pipeline.json';write(cfg,read(ROOT/'config/pipeline.json'))
            with patch('edgecode.workflow.model_metadata',return_value={'model':'fixture'}),patch('edgecode.workflow.generate',return_value=dict(answer='READY',generation_status='completed')):
                llm_check(cfg,argparse.Namespace())
            self.assertEqual(len(list((Path(d)/'outputs/llm').glob('check_*/generation_check.json'))),1)
