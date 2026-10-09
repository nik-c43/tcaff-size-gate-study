"""Read all original answers without author data, or inspect one task's provenance."""
import argparse
import html
import json
from pathlib import Path
from inspect_saved import ROOT, read

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task');p.add_argument('--html',type=Path)
    args=p.parse_args()
    labels=read(ROOT/'annotations/originals/labels-final-canonical.json')['labels']
    if args.task:
        row=next(r for r in labels if r['task_id']==args.task)
        meta=next(r for r in read(ROOT/'annotations/task_metadata.json')['tasks'] if r['task_id']==args.task)
        print(json.dumps(dict(answer=row,provenance=meta),ensure_ascii=False,indent=2))
    elif args.html:
        if args.html.exists():raise ValueError('Choose a new output file')
        rows=''.join('<tr>'+''.join('<td>'+html.escape(r[k])+'</td>' for k in ['task_id','label','confidence','notes'])+'</tr>' for r in labels)
        content='''<!doctype html><meta charset="utf-8"><title>Метки TCAFF</title>
<style>body{font:16px sans-serif;margin:2rem}td,th{padding:.5rem;text-align:left;border-bottom:1px solid #ddd}input{font:inherit;padding:.5rem;width:30rem}</style>
<h1>798 сохранённых ответов</h1><p>Изображения и маски здесь отсутствуют. Это чтение готовой разметки.</p>
<input aria-label="Поиск" placeholder="ID, метка, уверенность или комментарий" oninput="for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.innerText.toLowerCase().includes(this.value.toLowerCase())">
<table><thead><tr><th>ID</th><th>Метка</th><th>Уверенность</th><th>Комментарий</th></tr></thead><tbody>'''+rows+'</tbody></table>'
        args.html.write_text(content)
        print('Written:',args.html)
    else:
        for r in labels:print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()
