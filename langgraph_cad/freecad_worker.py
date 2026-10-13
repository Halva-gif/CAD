"""Measure the Python program written by the coding agent; no geometry templates."""
import json
import math
import os
from pathlib import Path
import runpy
import sys
import traceback


def metrics(name,shape):
    if shape.isNull():
        return {'object_name':name,'is_null':True,'is_valid':False,'solid_count':0,'volume':0.,'bbox':None}
    bb=shape.optimalBoundingBox(False,False)
    return {'object_name':name,'is_null':False,'is_valid':bool(shape.isValid()),
        'solid_count':len(shape.Solids),'volume':float(shape.Volume),
        'bbox':{'x_min':bb.XMin,'y_min':bb.YMin,'z_min':bb.ZMin,'x_max':bb.XMax,'y_max':bb.YMax,'z_max':bb.ZMax}}


def main():
    directory=Path(__file__).resolve().parent
    task=json.loads((directory/'task.json').read_text(encoding='utf-8'))
    report={'ok':False,'return_code':1,'fcstd_path':None,'error':None,'metrics':[],
        'screenshots':[],'step_path':None,'document_reopened':False,'step_roundtrip':False,
        'object_types':{},'parameters':{},'geometry_checks':[],
        'attempt_directory':str(directory),'failure_kind':None}
    doc=None; reopened=None; App=None; stage='loading_runtime'
    try:
        path=os.environ.get('CAD_FREECAD_LIB')
        if path:sys.path.insert(0,path)
        for candidate in ('/usr/lib/freecad-python3/lib','/usr/lib/freecad/lib'):
            if Path(candidate).is_dir():sys.path.append(candidate)
        import FreeCAD as App
        import Part
        # External Python embedding may point to a non-existent prefix/Mod.
        # Initialize paths from this installation's actual FreeCAD resources.
        mod_root=Path(os.environ.get('CAD_FREECAD_MOD_DIR',str(Path(App.getResourceDir())/'Mod')))
        if mod_root.is_dir():
            sys.path.append(str(mod_root))
            sys.path.extend(str(p) for p in sorted(mod_root.iterdir()) if p.is_dir())
        (directory/'runtime_loaded.txt').write_text('FreeCAD runtime loaded')
        stage='code'
        namespace=runpy.run_path(str(directory/'generated_code.py'),run_name='__main__')
        doc=App.ActiveDocument
        if doc is None:raise ValueError('Agent code created no FreeCAD document')
        doc.recompute()
        report['parameters']=namespace.get('CAD_PARAMETERS',{})
        for obj in doc.Objects:report['object_types'][obj.Name]=obj.TypeId
        final=task['script']['object_names']
        for name in final:
            obj=doc.getObject(name)
            if obj is None or not hasattr(obj,'Shape'):raise ValueError(f'Final object {name} is missing or has no Shape')
            report['metrics'].append(metrics(name,obj.Shape))
        part_shapes={item['part']:[doc.getObject(name).Shape for name in item['object_names']]
                     for item in task['script']['part_objects']}
        parameters={p['name']:p for p in task['cad_spec']['parameters']}
        for check in task['plan'].get('checks',[]):
            shapes=part_shapes[check['part']]
            shape=shapes[0] if len(shapes)==1 else Part.Compound(shapes)
            if check['metric']=='bbox':
                expected=check['expected']
                if check['parameter'] is not None:
                    p=parameters[check['parameter']]
                    aliases={'millimeter':'mm','millimeters':'mm','millimetre':'mm','millimetres':'mm','میلیمتر':'mm',
                        'centimeter':'cm','centimeters':'cm','centimetre':'cm','centimetres':'cm','سانتیمتر':'cm',
                        'meter':'m','meters':'m','metre':'m','metres':'m','متر':'m','inch':'in','inches':'in','اینچ':'in'}
                    unit=aliases.get(p['unit'].lower().replace(' ',''),p['unit'])
                    quantity=App.Units.Quantity(str(p['value'])+' '+unit)
                    expected=float(quantity.getValueAs('mm'))
                actual=getattr(shape.optimalBoundingBox(False,False),check['axis'].upper()+'Length')
                passed=abs(actual-expected)<=check['tolerance']
                detail=f'{actual:.8g} vs {expected:.8g} mm'
            elif check['metric']=='point_inside':
                actual=any(s.isInside(App.Vector(*check['point_mm']),1e-7,True) for s in shapes)
                passed=bool(actual)==check['inside']; detail=f'inside={actual}, expected={check["inside"]}'
            else:
                actual=sum(s.Volume for s in shapes)
                passed=check['minimum']<=actual<=check['maximum']; detail=f'Volume={actual:.8g} mm^3'
            report['geometry_checks'].append({'metric':check['metric'],'target':check['part'],'passed':bool(passed),'detail':check['reason']+': '+detail})
        fcstd=directory/'candidate.FCStd'; step=directory/'candidate.step'
        stage='artifacts'
        doc.saveAs(str(fcstd)); Part.export([doc.getObject(n) for n in final],str(step))
        original=report['metrics']
        App.closeDocument(doc.Name); doc=None
        reopened=App.openDocument(str(fcstd))
        restored=[metrics(n,reopened.getObject(n).Shape) for n in final]
        report['document_reopened']=all(a['is_valid'] and not a['is_null'] and a['solid_count']==b['solid_count']
            and abs(a['volume']-b['volume'])<=max(1e-5,b['volume']*1e-8)
            and all(abs(a['bbox'][k]-b['bbox'][k])<=1e-5 for k in a['bbox']) for a,b in zip(restored,original))
        imported=Part.read(str(step)); volume=sum(m['volume'] for m in original)
        report['step_roundtrip']=bool(imported.isValid() and not imported.isNull()
            and len(imported.Solids)==sum(m['solid_count'] for m in original)
            and abs(imported.Volume-volume)<=max(1e-4,volume*1e-7))
        report.update(fcstd_path=str(fcstd),step_path=str(step),return_code=0,
            ok=bool(report['document_reopened'] and report['step_roundtrip']))
    except Exception:
        report['error']=traceback.format_exc()[-6000:]
        report['failure_kind']='environment' if stage in ('loading_runtime','artifacts') else 'geometry'
    finally:
        if App is not None:
            for opened in (doc,reopened):
                if opened is not None:
                    try:App.closeDocument(opened.Name)
                    except Exception:pass
        (directory/'run_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return 0 if report['ok'] else 1


if __name__=='__main__':raise SystemExit(main())
