#!/usr/bin/env python3
"""Export development questions and their exact Click source spans for human audit."""
from pathlib import Path
import json, argparse
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--questions',type=Path,default=ROOT/'data/questions_dev20.json')
    parser.add_argument('--repo',type=Path,default=ROOT/'repository/click-8.1.7')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/development20/gold_audit.md')
    args=parser.parse_args()
    qs=json.loads(args.questions.read_text(encoding='utf8'))
    errors=[]; chunks=['# Gold evidence audit — Click 8.1.7','',
       'Review each source span for relevance to the question. A valid path/line bound does NOT guarantee semantic correctness.','']
    for q in qs:
        chunks.extend([f"## {q['id']} — {q.get('annotation_status','unverified')}",'',
                       f"**Question:** {q['question']}",'',f"**Expected answer (draft):** {q.get('expected_answer','')}",''])
        for gold in q['gold_spans']:
            path=gold['path']; start=int(gold['start_line']); end=int(gold['end_line']); full=(args.repo/path).resolve()
            if not full.is_relative_to(args.repo.resolve()) or not full.is_file():
                errors.append(f'{q["id"]}: missing/invalid file {path}');continue
            source=full.read_text(encoding='utf-8-sig').splitlines()
            if start<1 or end>len(source) or start>end:
                errors.append(f'{q["id"]}: span outside file {path} {start}-{end}');continue
            chunks.extend([f'**Evidence:** `{path}:{start}-{end}`','', '```python',
                           *[f'{i:>5}  {source[i-1]}' for i in range(start,end+1)],'```',''])
        chunks.extend(['**Manual review:** [ ] Confirm relevance  [ ] Correct range if necessary',''])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(chunks),encoding='utf-8')
    print(f'Wrote {len(qs)} annotated entries to: {args.output}')
    if errors:
        print('\n'.join(errors));raise SystemExit(1)
    print('All gold ranges exist within the repository, but SEMANTIC accuracy still requires human review.')

if __name__=='__main__':main()
