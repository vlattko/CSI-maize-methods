"""One explicit new-capture command, with separate inference and CPU fusion processes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from capture_fusion import sha


def stage_commands(python,raw,session,output,repo,card,provider,device,pixel_threshold=None,confidence=None):
    here=Path(__file__).resolve().parent
    capture=[python,str(here.parent/'new_session_pipeline_development_2026-10-27/run_capture.py'),'--input',str(raw),'--session',str(session),'--output',str(output/'capture'),'--repo',str(repo),'--model-card',str(card),'--provider',provider,'--device',device]
    if pixel_threshold is not None:capture+=['--pixel-threshold',str(pixel_threshold)]
    if confidence is not None:capture+=['--confidence',str(confidence)]
    fusion=[python,str(here/'capture_fusion.py'),'--capture',str(output/'capture'),'--raw',str(raw),'--repo',str(repo),'--output',str(output/'fusion')]
    return [capture,fusion]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','session','output','repo','model-card']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--provider',choices=['unet','yolo'],default='unet');p.add_argument('--device',default='cpu');p.add_argument('--pixel-threshold',type=float);p.add_argument('--confidence',type=float);p.add_argument('--plan',action='store_true');a=p.parse_args()
    raw=a.input.resolve();out=a.output.resolve()
    if out.is_relative_to(raw) or raw.is_relative_to(out):raise ValueError('Output must be separate from raw input')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise ValueError('Use a new or empty output folder')
    commands=stage_commands(sys.executable,raw,a.session.resolve(),out,a.repo.resolve(),a.model_card.resolve(),a.provider,a.device,a.pixel_threshold,a.confidence)
    if a.plan:print(json.dumps(dict(stages=commands,output=str(out)),indent=2));return
    out.mkdir(parents=True,exist_ok=True);record=dict(status='running',commands=commands,orchestrator_sha256=sha(Path(__file__)),session_sha256=sha(a.session),model_card_sha256=sha(a.model_card),stages=[]);started=time.time()
    environment={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
    def save():
        temporary=out/'digitization_manifest.tmp';temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(out/'digitization_manifest.json')
    save()
    for name,command in zip(['capture','fusion'],commands):
        stamp=time.time()
        with (out/(name+'.log')).open('w',encoding='utf-8') as stream:run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment)
        record['stages'].append(dict(stage=name,exit_code=run.returncode,seconds=time.time()-stamp));save()
        if run.returncode:record.update(status='failed',failed_stage=name);save();raise SystemExit(run.returncode)
        if name=='fusion':
            from format_report import format_report
            format_report(out/name/'report.html',update_manifest=True)
        record['stages'][-1]['output_manifest_sha256']=sha(out/name/'output_manifest.json');save()
    record.update(status='completed',elapsed_seconds=time.time()-started);save();print(json.dumps(record,indent=2))


if __name__=='__main__':main()
