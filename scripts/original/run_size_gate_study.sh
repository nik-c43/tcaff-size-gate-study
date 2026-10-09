#!/usr/bin/env bash
# Post-annotation entry point; do not start real experiments with partial labels.
set -euo pipefail
cd "$(dirname "$0")/.."

study_labels_path="${1:-data/size_gate_study/labels.json}"
study_run_root="${2:-data/size_gate_runs_$(date -u +%Y%m%dT%H%M%SZ)}"
study_count_path="data/size_gate_study/analysis_private/label_counts.json"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl

# Validate everything before creating the run directory or consuming real maps.
.venv/bin/python - "$study_labels_path" "$study_run_root" "$study_count_path" <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0,'scripts')
from size_gate_evaluate import validate_labels, annotation_counts
from size_gate_prepare import check_pins
from tcaff_baseline_audit import digest

try:
    labels_path,run_root,count_path=map(Path,sys.argv[1:])
    if not labels_path.is_file():
        raise ValueError('Нет файла разметки; сначала экспортируйте полный labels.json из интерфейса.')
    study=Path('data/size_gate_study')
    frozen=json.loads((study/'bundle_frozen.json').read_text())
    assert digest(study/'analysis_private/manifest.json')==frozen['manifest_sha256'],'Manifest изменился'
    assert digest(study/'annotation_blind/tasks.json')==frozen['blind_tasks_sha256'],'Список пар изменился'
    manifest=json.loads((study/'analysis_private/manifest.json').read_text())
    assert digest(Path(manifest['provenance']['configuration_path']))==manifest['provenance']['protocol_sha256'],'Протокол изменился'
    for file,expected in manifest['provenance']['code_hashes'].items():
        assert digest(Path(file))==expected,'Изменился замороженный код: '+file
    annotation=json.loads(labels_path.read_text())
    labels=validate_labels(manifest,annotation)
    check_pins(manifest['protocol'])
    if run_root.exists():
        raise ValueError('Каталог результатов уже существует; выберите новый второй аргумент.')
    if count_path.exists():
        counts=json.loads(count_path.read_text())
        assert counts['fingerprint']==manifest['fingerprint'],'Сохранённый подсчёт относится к другой выборке'
        assert counts['labels_sha256']==digest(labels_path),'Разметка изменилась после сохранённого подсчёта'
        expected=annotation_counts(manifest,labels)
        actual={key:counts.get(key) for key in expected}
        assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True),'Сохранённые подсчёты отличаются от разметки'
except (ValueError,AssertionError,KeyError,OSError) as exc:
    print('Запуск не выполнен: '+str(exc),file=sys.stderr)
    sys.exit(2)
PY

mkdir -p "$study_run_root"
.venv/bin/python - "$study_labels_path" "$study_run_root" <<'PY'
import json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from tcaff_baseline_audit import digest
label_path,run_root=map(Path,sys.argv[1:])
shutil.copyfile(label_path,run_root/'labels_snapshot.json')
metadata=dict(labels_source=str(label_path.resolve()),labels_sha256=digest(label_path),
              runner_sha256=digest(Path('scripts/run_size_gate_study.sh')),
              environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MPLCONFIGDIR']},
              stages=['count','development','holdout'],inputs='Frozen author maps; no changes to B0 or annotation package')
(run_root/'run_info.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
PY

# The copied export is the immutable annotation input for this entire run.
study_labels_snapshot="$study_run_root/labels_snapshot.json"
if [[ ! -e "$study_count_path" ]]; then
  .venv/bin/python scripts/size_gate_evaluate.py count \
    --labels "$study_labels_snapshot" --output "$study_count_path" > "$study_run_root/count.log" 2>&1
fi

echo "Development: $study_run_root/development.log"
.venv/bin/python scripts/size_gate_evaluate.py development \
  --labels "$study_labels_snapshot" --output "$study_run_root/development" > "$study_run_root/development.log" 2>&1

echo "Holdout: $study_run_root/holdout.log"
.venv/bin/python scripts/size_gate_evaluate.py holdout \
  --labels "$study_labels_snapshot" --selection "$study_run_root/development/selection.json" \
  --output "$study_run_root/holdout" > "$study_run_root/holdout.log" 2>&1

echo "Готово: $study_run_root/holdout/full_history_summary.json"
