"""Read-only integrity checks for the completed full-annotation study.

Writes a new verification record alongside the new results. Frozen inputs,
annotation assets, baseline files and upstream repositories are never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from size_gate_evaluate import validate_labels
from size_gate_prepare import check_pins


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='data/size_gate_runs_20261009')
    parser.add_argument('--results', default='results/size_gate_study_20261009')
    args = parser.parse_args()
    runs, results = ROOT / args.runs, ROOT / args.results
    target = results / 'final_checks.json'
    if target.exists() or not results.resolve().is_relative_to(ROOT / 'results'):
        raise ValueError('Use existing new results, without overwriting a verification record')
    summary = read(results / 'summary.json')
    manifest = read(ROOT / 'data/size_gate_study/analysis_private/manifest.json')
    frozen = read(ROOT / 'data/size_gate_study/bundle_frozen.json')
    assert digest(ROOT / 'data/size_gate_study/analysis_private/manifest.json') == frozen['manifest_sha256']
    assert digest(ROOT / manifest['provenance']['configuration_path']) == manifest['provenance']['protocol_sha256']
    for filename, expected in manifest['provenance']['code_hashes'].items():
        assert digest(ROOT / filename) == expected, filename
    for record in manifest['provenance']['frozen_B0_inputs']:
        assert digest(ROOT / record['path']) == record['sha256'], record['path']
    protected = read(ROOT / 'data/size_gate_preflight/protected_file_hashes.json')
    for filename, expected in protected.items():
        assert digest(ROOT / filename) == expected, filename
    check_pins(manifest['protocol'])
    labels = [ROOT/'data/labels(5).json', ROOT/'data/size_gate_study/labels.json', runs/'labels_snapshot.json']
    assert all(digest(path) == summary['labels_sha256'] for path in labels)
    answers = validate_labels(manifest, read(labels[-1]))
    assert len(answers) == 798 and summary['verified_run_files'] == 420
    assert summary['temporal_replay_verified_for_all_runs']
    for filename, expected in summary['provenance'].items():
        assert digest(ROOT / filename) == expected, filename
    assert digest(ROOT/'scripts/size_gate_results_report.py') == summary['report_script_sha256']
    assert digest(ROOT/'scripts/run_size_gate_study.sh') == read(runs/'run_info.json')['runner_sha256']
    upstream = {}
    for repo in ['vendor/tcaff', 'clipper']:
        upstream[repo] = dict(
            commit=subprocess.check_output(['git','-C',str(ROOT/repo),'rev-parse','HEAD'], text=True).strip(),
            tracked_status=subprocess.check_output(['git','-C',str(ROOT/repo),'status','--porcelain','--untracked-files=no'], text=True))
        assert upstream[repo]['tracked_status'] == ''
    output_files = sorted(p for p in results.rglob('*') if p.is_file())
    record = dict(all_checks_passed=True, protected_bundle_files_verified=len(protected),
                  frozen_code_files_verified=len(manifest['provenance']['code_hashes']),
                  baseline_result_files_verified=len(manifest['provenance']['frozen_B0_inputs']),
                  original_canonical_snapshot_labels_identical=True, complete_annotation_count=len(answers),
                  labels_sha256=summary['labels_sha256'], verified_run_files=summary['verified_run_files'],
                  temporal_replay_verified_for_all_runs=True, upstream=upstream,
                  output_hashes={str(p.relative_to(ROOT)):digest(p) for p in output_files},
                  verifier_sha256=digest(Path(__file__)), command=sys.argv,
                  scope='Integrity and parameter/replay records; not an independent statistical replication')
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['all_checks_passed','protected_bundle_files_verified',
                                           'baseline_result_files_verified','complete_annotation_count',
                                           'verified_run_files']},indent=2))


if __name__ == '__main__':
    main()
