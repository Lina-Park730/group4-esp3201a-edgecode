import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError
from edgecode.models import Chunk, SearchResult
from edgecode.generation import OllamaClient, OllamaError, answer_question, prepare_context, render_answer


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.results = [SearchResult(chunk=Chunk('src/example.py', 10, 11,
                        'def hello():\n    return 1', 'ast'), score=1, rank=1, source='dense')]

    def test_prompt_and_source_mapping(self):
        with patch('edgecode.generation.urlopen', return_value=io.BytesIO(
                json.dumps({'done': True, 'message': {'content': 'Returns 1 [S1].'}}).encode())) as request:
            result = answer_question(OllamaClient("test-model"), 'What is returned?', self.results)
        payload = json.loads(request.call_args.args[0].data)
        self.assertFalse(payload['stream'])
        self.assertIn('10: def hello()', payload['messages'][1]['content'])
        self.assertIn('src/example.py:10-11', render_answer(result))
        self.assertEqual(result['warnings'], [])

    def test_unknown_and_missing_citations(self):
        client = OllamaClient("test-model")
        for answer in ['Unknown [S99]', 'No references']:
            with patch.object(client, 'chat', return_value=answer):
                self.assertTrue(answer_question(client, 'question', self.results)['warnings'])

    def test_empty_context_does_not_call_model(self):
        client = OllamaClient("test-model")
        with patch.object(client, 'chat') as chat:
            result = answer_question(client, 'question', self.results, max_chars=1)
            chat.assert_not_called()
        self.assertEqual(result['sources'], [])

    def test_transport_errors(self):
        with patch('edgecode.generation.urlopen', side_effect=URLError('offline')):
            with self.assertRaises(OllamaError):
                OllamaClient("test-model").check()
        with patch('edgecode.generation.urlopen', return_value=io.BytesIO(b'not json')):
            with self.assertRaises(OllamaError):
                OllamaClient("test-model").check()

    def test_model_preflight(self):
        client = OllamaClient("test-model")
        with patch.object(client, 'request', side_effect=[{'version': 'test'}, {'models': []}]):
            with self.assertRaisesRegex(OllamaError, 'not installed'):
                client.check()

    def test_empty_answer(self):
        with patch('edgecode.generation.urlopen', return_value=io.BytesIO(b'{"message":{"content":""}}')):
            with self.assertRaisesRegex(OllamaError, 'no answer'):
                OllamaClient("test-model").chat([])
