import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI as RealChatOpenAI
import main
import test_pipeline
from schemas import RequirementSpec
from test_pipeline import data, FakeEmbeddings, AgentModel, tool_message


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

    def test_catalog_lookup_does_not_spend_embedding_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog=Path(directory)/'catalog.json';catalog.write_text(json.dumps({'sketch.md':'sketch geometry','pad.md':'pad extrusion'}))
            with patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'vector'}),patch.object(main,'get_embedding_model') as embedding:
                retriever=main.create_json_retriever(str(catalog))
                docs=retriever.invoke('pad')
            self.assertTrue(docs);self.assertEqual(docs[0].metadata['file_name'],'pad.md');embedding.assert_not_called()

    def test_followup_keeps_original_goal_in_context(self):
        context=main._request_context([{'role':'user','content':'a curved glass'},{'role':'user','content':'height 100 mm'}])
        self.assertIn('curved glass',context);self.assertIn('height 100 mm',context)

    def test_real_langchain_agent_uses_proxy_and_records_each_http_call(self):
        from contextlib import ExitStack
        req,spec,plan,script=data();captured=[];clients=[]
        def handler(request):
            body=json.loads(request.content);captured.append(body)
            return httpx.Response(200,json={'id':'code','object':'chat.completion','created':1,'model':'proxy-model',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':'',
                    'tool_calls':[{'id':'result','type':'function','function':{'name':'CodeGenerationResult',
                        'arguments':json.dumps({'status':'ready','script':script.model_dump(),'questions':[],'reason':''})}}]}}],
                'usage':{'prompt_tokens':1000,'completion_tokens':500,'total_tokens':1500}})
        def factory(**kwargs):
            client=httpx.Client(transport=httpx.MockTransport(handler));clients.append(client)
            return RealChatOpenAI(http_client=client,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            wiki,catalog,structured=test_pipeline.GraphTests().fixtures(d,script)
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ,{'OPENAI_BASE_URL':'https://proxy.example.test/v1',
                    'OPENAI_API_KEY':'test-key','CAD_MODEL':'proxy-model','CAD_RETRIEVAL_MODE':'lexical',
                    'CAD_INDEX_PATH':str(Path(d)/'index.sqlite')}))
                for name,value in [('model',None),('_knowledge_index',None),('wiki_directory',wiki),('json_path',str(catalog))]:
                    stack.enter_context(patch.object(main,name,value))
                stack.enter_context(patch.object(main,'ChatOpenAI',side_effect=factory))
                stack.enter_context(patch.object(main,'structured_model',side_effect=structured))
                result=main.build_graph().invoke({'messages':[{'role':'user','content':'Block 20 x 10 x 5 mm'}]})
        for client in clients:client.close()
        self.assertEqual(result['status'],'code_ready')
        self.assertEqual(result['generation_stats']['calls'][0]['input_tokens'],1000)
        self.assertEqual(result['generation_stats']['calls'][0]['output_tokens'],500)
        self.assertEqual(captured[0]['model'],'proxy-model')
        self.assertTrue(any(t['function']['name']=='CodeGenerationResult' for t in captured[0]['tools']))
        text='\n'.join(m['content'] for m in captured[0]['messages'])
        self.assertEqual(text.count('pad.Length = height'),1)

    def test_cli_saves_python_and_state_without_cad_execution(self):
        from contextlib import ExitStack
        req,spec,plan,script=data()
        model=AgentModel([tool_message('CodeGenerationResult',dict(status='ready',script=script.model_dump(),questions=[],reason=''),1)])
        with tempfile.TemporaryDirectory() as d:
            wiki,catalog,structured=test_pipeline.GraphTests().fixtures(d,script);out=Path(d)/'runs'
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'lexical','CAD_INDEX_PATH':str(Path(d)/'index.sqlite')}))
                for name,value in [('model',model),('_knowledge_index',None),('wiki_directory',wiki),('json_path',str(catalog))]:
                    stack.enter_context(patch.object(main,name,value))
                stack.enter_context(patch.object(main,'structured_model',side_effect=structured))
                stack.enter_context(patch('sys.argv',['main.py','--request','Block 20 x 10 x 5 mm','--out',str(out)]))
                self.assertEqual(main.main(),0)
            run=next(out.iterdir());state=json.loads((run/'state.json').read_text())
            self.assertEqual((run/'generated_code.py').read_text(),script.code)
            self.assertEqual(state['status'],'code_ready')
            self.assertFalse((run/'model.FCStd').exists())

    def test_index_only_calls_neither_chat_nor_codegen(self):
        with tempfile.TemporaryDirectory() as d:
            wiki=Path(d)/'wiki';wiki.mkdir();(wiki/'a.md').write_text('# A\nexample\n')
            with patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'lexical','CAD_INDEX_PATH':str(Path(d)/'index.sqlite')}),\
                 patch.object(main,'wiki_directory',wiki),patch.object(main,'_knowledge_index',None),\
                 patch.object(main,'get_model') as model,patch.object(main,'create_agent') as agent,\
                 patch('sys.argv',['main.py','--index-only']):
                self.assertEqual(main.main(),0);model.assert_not_called();agent.assert_not_called()
