import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

import main
from freecad_runner import run_script, validate_script
from knowledge_retrieval import CachedEmbeddings, pack_sections, split_markdown, safe_wiki_path, fenced_code
from model_validation import validate_model_result
from schemas import CadSpec, GeometryPlan, GeneratedScript, Parameter, RequirementSpec, RepairDecision
from schemas import ResolvedDimensions

FIXTURE=Path(__file__).parent/'fixtures'/'native_pad.py'


def data():
    parameters=[dict(name=n,value=v,unit='mm',status='given') for n,v in [('length',20),('width',10),('height',5)]]
    requirements=RequirementSpec(goal='Rectangular block',unit_system='mm',parameters=parameters,constraints=[],ambiguities=[])
    spec=CadSpec(parts=[dict(name='block',quantity=1,shape='rectangular block',features=[])],parameters=[dict(**p,part='block') for p in parameters],constraints=[],interfaces=[])
    plan=GeometryPlan(steps=[dict(id='block.profile',part='block',operation='sketch',depends_on=[],uses=[dict(source='parameter',ref=n) for n in ['length','width']],intent='Closed rectangle'),
        dict(id='block.pad',part='block',operation='pad',depends_on=['block.profile'],uses=[dict(source='parameter',ref='height'),dict(source='step',ref='block.profile')],intent='Pad the sketch to height')],
        checks=[dict(metric='bbox',part='block',axis=axis,parameter=name,reason='User dimension') for axis,name in [('x','length'),('y','width'),('z','height')]])
    script=GeneratedScript(code=FIXTURE.read_text(),object_names=['Pad'],part_objects=[dict(part='block',object_names=['Pad'])],
        step_objects=[dict(step_id='block.profile',object_names=['BaseSketch']),dict(step_id='block.pad',object_names=['Pad'])],knowledge_sources=['fixture.md'])
    return requirements,spec,plan,script


class FakeEmbeddings(Embeddings):
    def __init__(self):self.document_calls=0;self.query_calls=0
    def embed_documents(self,texts):self.document_calls+=1;return [[len(t),1.] for t in texts]
    def embed_query(self,text):self.query_calls+=1;return [len(text),1.]


class AgentModel(BaseChatModel):
    """Run real create_agent/tool/structured-output machinery without paid calls."""
    _responses: list = PrivateAttr()
    _seen: list = PrivateAttr(default_factory=list)
    def __init__(self,responses):super().__init__();self._responses=list(responses)
    @property
    def _llm_type(self):return 'test-agent'
    def bind_tools(self,tools,**kwargs):return self
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        self._seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self._responses.pop(0))])


def tool_message(name,args,index):
    return AIMessage(content='',tool_calls=[dict(name=name,args=args,id=f'call-{index}',type='tool_call')])


class Contracts(unittest.TestCase):
    def test_import_preserves_original_nodes_and_no_api_initialization(self):
        self.assertTrue({'normalize_request','specify_cad','plan_geometry','select_files','code_knowledge_extracter','code_generation'}<=set(main.graph.get_graph().nodes))

    def test_unknown_status_cannot_contain_an_invented_value(self):
        with self.assertRaises(ValueError):Parameter(name='height',value=100,unit='mm',status='unknown')

    def test_missing_unit_preserves_given_number_and_asks(self):
        req,spec,plan,script=data();req.parameters[0].unit=None
        out=main.resolve_parameters({'requirements':req.model_dump()})
        self.assertEqual(out['status'],'needs_clarification');self.assertEqual(out['requirements']['parameters'][0]['value'],20)

    def test_override_cannot_change_given_number(self):
        req,*_=data()
        with self.assertRaises(ValueError):main.resolve_parameters({'requirements':req.model_dump(),'parameter_overrides':{'height':{'value':9,'unit':'mm'}}})

    def test_assumptions_are_explicit_and_require_opt_in(self):
        req,*_=data();req.parameters[2]=Parameter(name='height',value=None,unit=None,status='unknown')
        chosen=req.model_copy(deep=True);chosen.parameters[2]=Parameter(name='height',value=5,unit='mm',status='assumed',assumption_reason='Prototype dimension explicitly authorized')
        class Extractor:
            def invoke(self,*args,**kwargs):return ResolvedDimensions(parameters=chosen.parameters)
        state={'requirements':req.model_dump(),'messages':[{'role':'user','content':'block'}],'allow_assumptions':True}
        with patch.object(main,'structured_model',return_value=Extractor()):out=main.resolve_parameters(state)
        self.assertEqual(out['requirements']['parameters'][2]['status'],'assumed')
        self.assertEqual(out['status'],'parameters_resolved')

    def test_assumption_resolver_cannot_change_given_facts(self):
        req,*_=data();req.parameters[2]=Parameter(name='height',value=None,unit=None,status='unknown')
        chosen=req.model_copy(deep=True);chosen.parameters[0].value=90
        chosen.parameters[2]=Parameter(name='height',value=5,unit='mm',status='assumed',assumption_reason='prototype')
        class Extractor:
            def invoke(self,*args,**kwargs):return ResolvedDimensions(parameters=chosen.parameters)
        with patch.object(main,'structured_model',return_value=Extractor()),self.assertRaises(ValueError):
            main.resolve_parameters({'requirements':req.model_dump(),'messages':[{'role':'user','content':'block'}],'allow_assumptions':True})

    def test_rounding_radius_is_a_valid_measurable_size(self):
        req,*_=data();req.parameters[0].name='rounding_radius'
        self.assertIsNone(main._requirement_gap(req))

    def test_normalizer_cannot_make_assumptions_on_its_own(self):
        req,*_=data();req.parameters[0].status='assumed';req.parameters[0].assumption_reason='invented'
        self.assertIsNotNone(main._requirement_gap(req))

    def test_missing_metadata_is_repairable_contract_failure(self):
        req,spec,plan,script=data();script.step_objects=[]
        with tempfile.TemporaryDirectory() as directory:
            run=run_script(script,spec,plan,directory)
            self.assertFalse(run.ok);self.assertIn('step_objects',run.error)

    def test_unread_source_is_rejected(self):
        req,spec,plan,script=data()
        with self.assertRaises(ValueError):validate_script(script,spec,plan,known_sources=['different.md'])

    def test_dynamic_execution_and_io_are_rejected(self):
        req,spec,plan,script=data()
        for code in ["import os\nos.system('id')", "exec('x=1')", "open('/etc/passwd')"]:
            bad=script.model_copy(deep=True);bad.code=code
            with self.subTest(code=code),self.assertRaises(ValueError):validate_script(bad,spec,plan)

    def test_no_false_accept_from_repair_agent(self):
        req,spec,plan,script=data()
        class Extractor:
            def invoke(self,*args,**kwargs):return RepairDecision(action='accept',reason='pretend',step_ids=[])
        state={'cad_spec':spec.model_dump(),'geometry_plan':plan.model_dump(),'script':script.model_dump(),
            'validation':{'passed':False,'checks':[]},'run_result':{'failure_kind':'geometry'},'attempt':0}
        with patch.object(main,'structured_model',return_value=Extractor()),self.assertRaises(ValueError):main.repair_model(state)


class Retrieval(unittest.TestCase):
    def test_freecad_wiki_code_templates_are_preserved(self):
        self.assertIn('doc = App.newDocument()',fenced_code('{{Code|code=doc = App.newDocument()}}'))
        self.assertIn('x = 1',fenced_code('<syntaxhighlight lang="python">x = 1</syntaxhighlight>'))
        docs=split_markdown('# Example\n{{Code|code=\n# Python comment\nx = 1\n}}\n')
        self.assertEqual(len(docs),1)
        self.assertIn('# Python comment',fenced_code(docs[0].page_content))
    def test_sections_after_twelfth_are_kept_with_complete_code(self):
        text=''.join(f'## Section {i}\nprose\n' for i in range(20))+'## Code\n```python\n# heading inside code\nx = 1\n```\n'
        docs=split_markdown(text,'fixture.md')
        self.assertEqual(len(docs),21)
        self.assertIn('x = 1',docs[-1].page_content)
        self.assertIn('```',docs[-1].page_content)

    def test_budget_does_not_cut_code_example(self):
        docs=split_markdown('# Example\n```python\n'+('x = 1\n'*50)+'```')
        self.assertEqual(pack_sections(docs,50),[])

    def test_embedding_cache_reused_and_invalidated_by_content(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapped=FakeEmbeddings();path=Path(directory)/'cache.sqlite'
            a=CachedEmbeddings(wrapped,path,'model-a');a.embed_documents(['doc','doc']);a.embed_query('query')
            b=CachedEmbeddings(wrapped,path,'model-a');b.embed_documents(['doc']);b.embed_query('query')
            self.assertEqual((wrapped.document_calls,wrapped.query_calls),(1,1))
            b.embed_documents(['changed']);self.assertEqual(wrapped.document_calls,2)

    def test_paths_cannot_escape_knowledge_directory(self):
        with self.assertRaises(ValueError):safe_wiki_path('/tmp','../secret.md')


class GraphTests(unittest.TestCase):
    def fixtures(self,directory,script):
        root=Path(directory);wiki=root/'wiki';wiki.mkdir()
        (wiki/'fixture.md').write_text('# Native FreeCAD sketch and pad\n```python\n'+FIXTURE.read_text()+'\n```\n')
        catalog=root/'catalog.json';catalog.write_text(json.dumps({'fixture.md':'sketch pad FreeCAD document examples'}))
        req,spec,plan,_=data()
        class Extractor:
            def __init__(self,value):self.value=value
            def invoke(self,*args,**kwargs):return self.value
        def structured(schema):
            values={RequirementSpec:req,CadSpec:spec,GeometryPlan:plan,main.DocNames:main.DocNames(selected_files=[dict(file_name='fixture.md',read='Native sketch and pad')]),RepairDecision:RepairDecision(action='revise_code',reason='Fix actual dimensional mismatch',step_ids=[])}
            return Extractor(values[schema])
        return wiki,catalog,structured

    def test_missing_units_stops_before_specification_and_codegen(self):
        req,*_=data();req.parameters[0].unit=None
        class Extractor:
            def invoke(self,*args,**kwargs):return req
        with patch.object(main,'structured_model',return_value=Extractor()) as model:
            result=main.build_graph().invoke({'messages':[{'role':'user','content':'Block'}]})
        self.assertEqual(result['status'],'needs_clarification');self.assertEqual(model.call_count,1)

    def test_missing_document_stops_without_codegen_fallback(self):
        req,spec,plan,script=data()
        with tempfile.TemporaryDirectory() as directory:
            wiki,catalog,structured=self.fixtures(directory,script);(wiki/'fixture.md').unlink()
            with patch.object(main,'wiki_directory',wiki),patch.object(main,'json_path',str(catalog)),patch.object(main,'_json_retriever',None),patch.object(main,'_md_retrievers',{}),patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'lexical'}),patch.object(main,'structured_model',side_effect=structured),patch.object(main,'create_agent') as agent:
                result=main.build_graph().invoke({'messages':[{'role':'user','content':'Block 20 x 10 x 5 mm'}]})
                self.assertEqual(result['status'],'knowledge_insufficient');agent.assert_not_called()

    def test_actual_agent_tools_and_structured_output_receive_full_context(self):
        req,spec,plan,script=data()
        responses=[tool_message('read_knowledge',dict(file_name='fixture.md',query='sketch pad'),1),
            tool_message('CodeGenerationResult',dict(status='ready',script=script.model_dump(),questions=[],reason=''),2)]
        agent_model=AgentModel(responses)
        with tempfile.TemporaryDirectory() as directory:
            wiki,catalog,structured=self.fixtures(directory,script)
            with patch.object(main,'wiki_directory',wiki),patch.object(main,'json_path',str(catalog)),patch.object(main,'_json_retriever',None),patch.object(main,'_md_retrievers',{}),patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'lexical'}),patch.object(main,'structured_model',side_effect=structured),patch.object(main,'model',agent_model):
                result=main.build_graph().invoke({'messages':[{'role':'user','content':'Block 20 x 10 x 5 mm'}],'execute':False})
            self.assertEqual(result['status'],'code_ready');self.assertEqual(result['generated_code'],script.code)
            payload=json.loads(agent_model._seen[0][-1].content)
            self.assertEqual(payload['cad_spec']['parts'][0]['name'],'block')
            self.assertEqual(payload['geometry_plan']['steps'][1]['operation'],'pad')
            self.assertIn('pad.Length',payload['knowledge'][0]['note'])
            self.assertTrue(any(m.type=='tool' for m in agent_model._seen[1]))


@unittest.skipUnless(os.environ.get('CAD_RUN_INTEGRATION')=='1','Enable real FreeCAD tests with CAD_RUN_INTEGRATION=1')
class RealFreeCAD(unittest.TestCase):
    def test_agent_program_creates_native_sketch_and_pad(self):
        req,spec,plan,script=data()
        with tempfile.TemporaryDirectory() as directory:
            run=run_script(script,spec,plan,directory,known_sources=['fixture.md'])
            self.assertTrue(run.ok,run.error)
            report=validate_model_result(spec,plan,script,run)
            self.assertTrue(report.passed,report.model_dump())
            self.assertEqual(run.object_types['BaseSketch'],'Sketcher::SketchObject')
            self.assertEqual(run.object_types['Pad'],'PartDesign::Pad')
            self.assertAlmostEqual(run.metrics[0].volume,1000)

    def test_wrong_height_fails_actual_geometry_even_if_metadata_claims_right_value(self):
        req,spec,plan,script=data();script.code=script.code.replace('pad.Length = height','pad.Length = height/2')
        with tempfile.TemporaryDirectory() as directory:
            run=run_script(script,spec,plan,directory)
            report=validate_model_result(spec,plan,script,run)
            self.assertFalse(report.passed)
            self.assertTrue(any(c.metric=='bbox' and not c.passed for c in report.checks))

    def test_original_bad_profile_is_not_accepted(self):
        req,spec,plan,script=data()
        script.code=(Path(__file__).resolve().parents[1]/'original/cad_testing.py').read_text()
        with tempfile.TemporaryDirectory() as directory:
            run=run_script(script,spec,plan,directory)
            self.assertFalse(run.ok);self.assertIn('no FreeCAD document',run.error)

    def test_real_langgraph_repair_loop_with_real_agent_and_freecad(self):
        req,spec,plan,good=data();bad=good.model_copy(deep=True);bad.code=bad.code.replace('pad.Length = height','pad.Length = height/2')
        agent_model=AgentModel([tool_message('CodeGenerationResult',dict(status='ready',script=s.model_dump(),questions=[],reason=''),i) for i,s in enumerate([bad,good],1)])
        fixtures=GraphTests()
        with tempfile.TemporaryDirectory() as directory:
            wiki,catalog,structured=fixtures.fixtures(directory,good)
            with patch.object(main,'wiki_directory',wiki),patch.object(main,'json_path',str(catalog)),patch.object(main,'_json_retriever',None),patch.object(main,'_md_retrievers',{}),patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'lexical'}),patch.object(main,'structured_model',side_effect=structured),patch.object(main,'model',agent_model):
                result=main.build_graph().invoke({'messages':[{'role':'user','content':'Block 20 x 10 x 5 mm'}],'output_directory':directory},config={'recursion_limit':80})
            self.assertEqual(result['status'],'validated',result.get('run_result'))
            self.assertEqual(result['attempt'],1)
            self.assertTrue((Path(directory)/'model.FCStd').is_file())
            second=json.loads(agent_model._seen[1][-1].content)
            self.assertFalse(second['validation']['passed'])
            self.assertEqual(second['previous_script']['code'],bad.code)

    def test_repair_budget_stops_and_does_not_publish_bad_model(self):
        req,spec,plan,bad=data();bad.code=bad.code.replace('pad.Length = height','pad.Length = height/2')
        agent_model=AgentModel([tool_message('CodeGenerationResult',dict(status='ready',script=bad.model_dump(),questions=[],reason=''),i) for i in range(4)])
        fixtures=GraphTests()
        with tempfile.TemporaryDirectory() as directory:
            wiki,catalog,structured=fixtures.fixtures(directory,bad)
            with patch.object(main,'wiki_directory',wiki),patch.object(main,'json_path',str(catalog)),patch.object(main,'_json_retriever',None),patch.object(main,'_md_retrievers',{}),patch.dict(os.environ,{'CAD_RETRIEVAL_MODE':'lexical'}),patch.object(main,'structured_model',side_effect=structured),patch.object(main,'model',agent_model):
                result=main.build_graph().invoke({'messages':[{'role':'user','content':'block'}],'output_directory':directory},config={'recursion_limit':80})
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['attempt'],3)
            self.assertEqual(len(agent_model._seen),4)
            self.assertFalse((Path(directory)/'model.FCStd').exists())
