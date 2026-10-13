"""Execute actual agent-written Python with a fresh worker and bounded runtime.

AST checks and a process boundary are not a complete security sandbox. Production
execution of generated Python should use a disposable container/worker isolation.
"""
import ast
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile

from schemas import RunResult


def validate_script(script,spec,plan,known_sources=None,allow_assumptions=False):
    if not script.code.strip() or len(script.code)>200000:raise ValueError('script is empty or too large')
    tree=ast.parse(script.code)
    blocked_modules={'os','sys','subprocess','pathlib','socket','requests','urllib','http','ftplib','shutil',
        'pickle','marshal','ctypes','importlib','builtins','multiprocessing','threading','resource','signal','io'}
    blocked_calls={'open','exec','eval','compile','__import__','input','exit','quit'}
    blocked_attributes={'system','popen','saveAs','save','exportStep','exportStl','exportBrep','export','openDocument'}
    for node in ast.walk(tree):
        modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        if any(name.split('.')[0] in blocked_modules for name in modules):raise ValueError('model code cannot import filesystem/network/process modules')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in blocked_calls:raise ValueError('dynamic execution or external IO in model code is prohibited')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('__') or node.attr in blocked_attributes):raise ValueError('runner owns IO/export; no introspection or process access in model code')
    parts={p.name for p in spec.parts}; ids={s.id for s in plan.steps}
    if {p.part for p in script.part_objects}!=parts or len(script.part_objects)!=len(parts):raise ValueError('part_objects must map each exact CAD part once')
    final=[name for p in script.part_objects for name in p.object_names]
    if not final or len(set(final))!=len(final) or set(final)!=set(script.object_names) or len(set(script.object_names))!=len(script.object_names):raise ValueError('final object_names must exactly match unique part_objects; exclude construction tools')
    if {s.step_id for s in script.step_objects}!=ids or len(script.step_objects)!=len(ids) or any(not s.object_names for s in script.step_objects):raise ValueError('step_objects must map every plan step to real objects')
    if known_sources is not None and (not script.knowledge_sources or set(script.knowledge_sources)-set(known_sources)):
        raise ValueError('code cites unread or missing knowledge sources')
    if script.assumptions and not allow_assumptions:raise ValueError('dimensional assumptions require explicit allow_assumptions')
    if any(p.status!='assumed' for p in script.assumptions):raise ValueError('additional dimensional choices must be marked assumed')
    names=[p.name for p in spec.parameters]+[p.name for p in script.assumptions]
    if len(names)!=len(set(names)):raise ValueError('assumptions duplicate or shadow known dimensions')


def runtime_command():
    configured=os.environ.get('CAD_FREECAD_COMMAND')
    if configured:
        argv=json.loads(configured)
        if not isinstance(argv,list) or not argv or any(not isinstance(v,str) for v in argv):raise ValueError('CAD_FREECAD_COMMAND must be a JSON argv list')
        return argv
    if Path('/usr/lib/freecad-python3/lib/FreeCAD.so').is_file():return ['/usr/bin/python3']
    for name in ('FreeCADCmd','freecadcmd'):
        found=shutil.which(name)
        if found:return [found]
    raise FileNotFoundError('Install/configure FreeCAD runtime with CAD_FREECAD_COMMAND and CAD_FREECAD_LIB')


def run_script(script,spec,plan,output_directory,known_sources=None,allow_assumptions=False):
    root=Path(output_directory).resolve(); root.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='attempt-',dir=root))
    (directory/'generated_code.py').write_text(script.code,encoding='utf-8')
    failure={'ok':False,'return_code':1,'fcstd_path':None,'error':None,'metrics':[],'screenshots':[],
             'attempt_directory':str(directory),'failure_kind':'geometry'}
    try:validate_script(script,spec,plan,known_sources,allow_assumptions)
    except (ValueError,SyntaxError) as exc:
        failure['error']=f'Generated script contract failed: {exc}'; return RunResult.model_validate(failure)
    (directory/'task.json').write_text(json.dumps({'script':script.model_dump(),'cad_spec':spec.model_dump(),'plan':plan.model_dump()},ensure_ascii=False),encoding='utf-8')
    worker=directory/'freecad_worker.py'; shutil.copy2(Path(__file__).with_name('freecad_worker.py'),worker)
    try:
        command=runtime_command()
        allowed=('PATH','LD_LIBRARY_PATH','SystemRoot','WINDIR','LANG','LC_ALL','CAD_FREECAD_LIB','CAD_FREECAD_MOD_DIR','QT_QPA_PLATFORM','QT_PLUGIN_PATH')
        environment={k:os.environ[k] for k in allowed if k in os.environ}
        environment['PYTHONNOUSERSITE']='1'
        timeout=float(os.environ.get('CAD_EXECUTION_TIMEOUT','90'))
        with (directory/'stdout.log').open('wb') as out,(directory/'stderr.log').open('wb') as err:
            process=subprocess.Popen([*command,str(worker)],cwd=directory,env=environment,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=os.name=='posix')
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait(); failure.update(error=f'FreeCAD exceeded {timeout}s',failure_kind='timeout')
        report=directory/'run_result.json'
        if report.is_file() and failure['failure_kind']!='timeout':
            result=RunResult.model_validate(json.loads(report.read_text(encoding='utf-8')))
            result.return_code=process.returncode; result.ok=result.ok and process.returncode==0
            return result
        if not failure['error']:
            tails=[]
            for file in ('stdout.log','stderr.log'):
                with (directory/file).open('rb') as stream:
                    stream.seek(0,2); stream.seek(max(0,stream.tell()-3000)); tails.append(stream.read().decode(errors='replace'))
            failure.update(error='Worker produced no report. '+'\n'.join(tails),
                failure_kind='geometry' if (directory/'runtime_loaded.txt').exists() else 'environment')
        failure['return_code']=process.returncode
    except (OSError,ValueError) as exc:failure.update(error=str(exc),failure_kind='environment')
    return RunResult.model_validate(failure)
