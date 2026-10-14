from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
import test_pipeline
from schemas import Parameter, RequirementSpec
from test_pipeline import AgentModel, data, tool_message


class AgentBudgets(unittest.TestCase):
    def run_case(self,responses,settings=None,extra=None):
        req,spec,plan,script=data()
        model=AgentModel(responses)
        with tempfile.TemporaryDirectory() as d:
            wiki,catalog,structured=test_pipeline.GraphTests().fixtures(d,script)
            if extra:
                for name,body in extra.items():(wiki/name).write_text(body)
            env={'CAD_RETRIEVAL_MODE':'lexical','CAD_INDEX_PATH':str(Path(d)/'index.sqlite'),
                 'CAD_AGENT_MAX_MODEL_CALLS':'6','CAD_AGENT_MAX_INPUT_TOKENS':'14000',
                 'CAD_AGENT_MAX_TOTAL_INPUT_TOKENS':'60000','CAD_AGENT_MAX_TOOL_CALLS':'6'}
            env.update(settings or {})
            with ExitStack() as stack:
                for name,value in [('wiki_directory',wiki),('json_path',str(catalog)),('_knowledge_index',None),
                                   ('model',model)]:stack.enter_context(patch.object(main,name,value))
                stack.enter_context(patch.dict(os.environ,env))
                stack.enter_context(patch.object(main,'structured_model',side_effect=structured))
                result=main.build_graph().invoke({'messages':[{'role':'user','content':'Block 20 x 10 x 5 mm'}]})
            return result,model

    def ready(self,script,index):
        return tool_message('CodeGenerationResult',dict(status='ready',script=script.model_dump(),questions=[],reason=''),index)

    def test_additional_source_is_loaded_once_and_tool_receipts_are_small(self):
        req,spec,plan,script=data();script.knowledge_sources.append('extra.md')
        responses=[tool_message('read_knowledge',dict(file_name='extra.md',query='uniqueAPI'),1),
                   tool_message('read_knowledge',dict(file_name='extra.md',query='uniqueAPI'),2),self.ready(script,3)]
        result,model=self.run_case(responses,extra={'extra.md':'# Unique API\n```python\nuniqueAPI()\n```\n'})
        self.assertEqual(result['status'],'code_ready')
        self.assertEqual(result['generation_stats']['model_calls'],3)
        second=json.loads(model._seen[1][1].content)
        third=json.loads(model._seen[2][1].content)
        self.assertEqual(len(second['knowledge']),2)
        self.assertEqual(second['knowledge'],third['knowledge'])
        for messages in model._seen[1:]:
            text='\n'.join(str(m.content) for m in messages)
            self.assertEqual(text.count('uniqueAPI()'),1)
            self.assertTrue(all('uniqueAPI()' not in str(m.content) for m in messages if m.type=='tool'))

    def test_model_call_cap_stops_before_another_paid_call(self):
        responses=[tool_message('read_knowledge',dict(file_name='fixture.md',query='sketch'),i) for i in range(3)]
        result,model=self.run_case(responses,{'CAD_AGENT_MAX_MODEL_CALLS':'2'})
        self.assertEqual(result['status'],'token_budget_exceeded')
        self.assertEqual(len(model._seen),2)
        self.assertTrue(result['errors'])

    def test_context_cap_stops_before_first_paid_call(self):
        result,model=self.run_case([],{'CAD_AGENT_MAX_INPUT_TOKENS':'100'})
        self.assertEqual(result['status'],'token_budget_exceeded')
        self.assertEqual(len(model._seen),0)

    def test_cumulative_input_cap_stops_repeated_calls(self):
        responses=[tool_message('read_knowledge',dict(file_name='fixture.md',query='sketch'),i) for i in range(3)]
        result,model=self.run_case(responses,{'CAD_AGENT_MAX_TOTAL_INPUT_TOKENS':'15000'})
        self.assertEqual(result['status'],'token_budget_exceeded')
        self.assertEqual(len(model._seen),1)

    def test_tool_cap_does_not_load_additional_documents(self):
        req,spec,plan,script=data()
        responses=[tool_message('search_knowledge',dict(query='uniqueAPI'),1),
                   tool_message('read_knowledge',dict(file_name='extra.md',query='uniqueAPI'),2),self.ready(script,3)]
        result,model=self.run_case(responses,{'CAD_AGENT_MAX_TOOL_CALLS':'1'},
            extra={'extra.md':'# Extra\n```python\nuniqueAPI()\n```\n'})
        self.assertEqual(result['status'],'code_ready')
        self.assertEqual({d['file_name'] for d in result['file_knowledge']},{'fixture.md'})
        self.assertTrue(any('budget exhausted' in str(m.content).lower() for m in model._seen[-1] if m.type=='tool'))

    def test_unread_citation_is_reported_without_executing_or_retrying(self):
        req,spec,plan,script=data();script.knowledge_sources=['unread.md']
        result,model=self.run_case([self.ready(script,1)])
        self.assertEqual(result['status'],'code_invalid')
        self.assertEqual(len(model._seen),1)
        self.assertIn('unread',result['errors'][0])

    def test_invalid_python_is_reported_without_runtime_or_retry(self):
        req,spec,plan,script=data();script.code='def broken('
        result,model=self.run_case([self.ready(script,1)])
        self.assertEqual(result['status'],'code_invalid');self.assertEqual(len(model._seen),1)

    def test_graph_ends_at_code_generation_and_has_no_runtime_nodes(self):
        graph=main.build_graph().get_graph()
        self.assertFalse({'run_freecad','validate_model','repair_model'} & set(graph.nodes))
        self.assertTrue(any(e.source=='code_generation' and e.target=='__end__' for e in graph.edges))

    def test_unknown_dimension_duplicates_are_removed(self):
        req=RequirementSpec(goal='Glass',unit_system=None,parameters=[
            Parameter(name='rim_diameter',value=None,unit=None,status='unknown')],constraints=[],
            ambiguities=['The rim diameter is unspecified.','rim_diameter is unspecified'])
        main._record_unknowns(req)
        self.assertEqual(len(req.ambiguities),1)

    def test_initial_knowledge_does_not_starve_later_operations(self):
        from langchain_core.documents import Document
        from knowledge_retrieval import split_markdown
        docs={}
        for name,body in [('a','# sketch\n'+'a'*1200),('b','# pad\n'+'b'*1600),
                          ('c','# pad small\n'+'c'*300)]:
            doc=split_markdown(body,name+'.md')[0];docs[name]=doc
        class Index:
            counter=main.TokenCounter()
            def section(self,key):return docs[key]
        req,spec,plan,script=data()
        with patch.object(main,'get_knowledge_index',return_value=Index()),patch.dict(os.environ,{
            'CAD_KNOWLEDGE_INITIAL_TOKENS':'2100','CAD_KNOWLEDGE_TOTAL_TOKENS':'5000'}):
            result=main.code_knowledge_extracter({'geometry_plan':plan.model_dump(),
                'selection_candidates':{'sketch':['a'],'pad':['b','c']}})
        self.assertEqual(result['status'],'knowledge_ready')
        self.assertEqual({s['file_name'] for s in result['file_knowledge']},{'a.md','c.md'})
