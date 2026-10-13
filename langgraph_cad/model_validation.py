"""Deterministic validation of the agent's executed FreeCAD program."""
import math
from pathlib import Path

from schemas import CheckResult, RunResult, ValidationReport


def validate_model_result(spec,plan,script,run):
    run=RunResult.model_validate(run); checks=[]
    def check(metric,target,passed,detail):
        checks.append(CheckResult(metric=metric,target=target,passed=bool(passed),detail=detail))
    check('document_saved','FreeCAD',run.ok and run.return_code==0,run.error or 'Worker completed')
    for path,label in ((run.fcstd_path,'FCStd'),(run.step_path,'STEP')):
        valid=bool(path and run.attempt_directory and Path(path).resolve().is_relative_to(Path(run.attempt_directory).resolve()) and Path(path).is_file() and Path(path).stat().st_size>0)
        check('document_saved',label,valid,'Nonempty artifact created in the fresh attempt directory')
    check('document_saved','FCStd roundtrip',run.document_reopened,'Reopened document matches final shapes')
    check('document_saved','STEP roundtrip',run.step_roundtrip,'Imported STEP preserves volume and solid count')
    metrics={m.object_name:m for m in run.metrics}
    check('feature_present','final objects',set(metrics)==set(script.object_names) and len(metrics)==len(run.metrics),'Exact final object set')
    for name in script.object_names:
        m=metrics.get(name)
        check('shape_valid',name,m is not None and not m.is_null and m.is_valid,'Non-null valid shape')
        check('volume_positive',name,m is not None and m.volume is not None and math.isfinite(m.volume) and m.volume>0,'Actual positive volume')
    for part in spec.parts:
        mapping=next((p for p in script.part_objects if p.part==part.name),None)
        count=sum(metrics[n].solid_count for n in mapping.object_names if n in metrics) if mapping else 0
        check('solid_count',part.name,count==part.quantity,f'Expected {part.quantity} solids; measured {count}')
    for step in script.step_objects:
        check('feature_present',step.step_id,all(name in run.object_types for name in step.object_names),'Mapped construction objects exist in the document')
    for parameter in [*spec.parameters,*script.assumptions]:
        actual=run.parameters.get(parameter.name)
        correct=isinstance(actual,dict) and type(actual.get('value')) in (int,float) and actual.get('unit')==parameter.unit and math.isclose(actual['value'],parameter.value,rel_tol=1e-10,abs_tol=1e-10)
        check('parameter',parameter.name,correct,'Executed CAD_PARAMETERS preserves the recorded value and unit; geometry is checked separately')
    check('bbox','acceptance coverage',len(run.geometry_checks)==len(plan.checks),'Every pre-generation geometric check was executed')
    for item in run.geometry_checks:
        check(item['metric'],item['target'],item['passed'] is True,item['detail'])
    measured={c.parameter for c in plan.checks if c.parameter is not None}
    unmeasured=[f'Dimension {p.name}: no direct geometric measurement tied to this parameter; metadata audit alone is not a measurement.' for p in spec.parameters if p.name not in measured]
    return ValidationReport(passed=all(c.passed for c in checks),checks=checks,
        unverified_constraints=list(spec.constraints)+([i.relation for i in spec.interfaces])+unmeasured)
