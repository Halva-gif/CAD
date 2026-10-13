import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI as RealChatOpenAI
import main
from schemas import RequirementSpec
from test_pipeline import data, FakeEmbeddings


class Configuration(unittest.TestCase):
    def test_langchain_structured_model_uses_proxy_base_url_and_function_calling(self):
        captured=[];clients=[]
        req,*_=data()
        def handler(request):
            captured.append(request)
            return httpx.Response(200,json={'id':'test','object':'chat.completion','created':1,'model':'proxy-model',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':'',
                    'tool_calls':[{'id':'r1','type':'function','function':{'name':'RequirementSpec','arguments':json.dumps(req.model_dump())}}]}}],
                'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}})
        def factory(**kwargs):
            transport=httpx.Client(transport=httpx.MockTransport(handler));clients.append(transport)
            return RealChatOpenAI(http_client=transport,**kwargs)
        with patch.dict(os.environ,{'OPENAI_BASE_URL':'https://proxy.example.test/v1','OPENAI_API_KEY':'test-key','CAD_MODEL':'proxy-model','CAD_STRUCTURED_METHOD':'function_calling'}),patch.object(main,'model',None),patch.object(main,'ChatOpenAI',side_effect=factory):
            result=main.structured_model(RequirementSpec).invoke('block request')
        for client in clients:client.close()
        self.assertEqual(result.goal,'Rectangular block')
        self.assertEqual(str(captured[0].url),'https://proxy.example.test/v1/chat/completions')
        body=json.loads(captured[0].content)
        self.assertEqual(body['model'],'proxy-model')
        self.assertTrue(any(t['function']['name']=='RequirementSpec' for t in body['tools']))

    def test_embeddings_can_use_a_different_provider_from_chat(self):
        captured=[]
        def factory(**kwargs):captured.append(kwargs);return FakeEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main,'ROOT',Path(directory)),patch.object(main,'json_embedding_model',None),patch.object(main,'OpenAIEmbeddings',side_effect=factory),patch.dict(os.environ,{
                'OPENAI_BASE_URL':'https://chat.example.test/v1','OPENAI_API_KEY':'test-chat-key',
                'OPENAI_EMBEDDINGS_BASE_URL':'https://embedding.example.test/v1','OPENAI_EMBEDDINGS_API_KEY':'test-embedding-key','CAD_EMBEDDING_MODEL':'embedding-model'}):
                main.get_embedding_model().embed_documents(['source'])
        self.assertEqual(captured[0]['base_url'],'https://embedding.example.test/v1')
        self.assertEqual(captured[0]['model'],'embedding-model')

    def test_original_vector_retrieval_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog=Path(directory)/'catalog.json';catalog.write_text(json.dumps({'sketch.md':'sketch geometry','pad.md':'pad extrusion'}))
            with patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'vector'}),patch.object(main,'get_embedding_model',return_value=FakeEmbeddings()):
                retriever=main.create_json_retriever(str(catalog))
                docs=retriever.invoke('pad')
            self.assertTrue(docs);self.assertIn(docs[0].metadata['file_name'],['sketch.md','pad.md'])

    def test_followup_keeps_original_goal_in_context(self):
        context=main._request_context([{'role':'user','content':'a curved glass'},{'role':'user','content':'height 100 mm'}])
        self.assertIn('curved glass',context);self.assertIn('height 100 mm',context)
