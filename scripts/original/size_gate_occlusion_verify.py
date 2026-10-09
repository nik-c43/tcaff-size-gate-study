"""Verify the added occlusion model and the preserved earlier research snapshot."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from size_gate_candidate_stage_audit import read, digest
from size_gate_prepare import check_pins, gate_details


def main():
    out = ROOT / 'results/size_gate_research_continuation_20261009'
    if out.exists():
        raise ValueError('Refuse overwrite')
    previous_path = ROOT / 'results/size_gate_research_20261009/final_checks.json'
    previous = read(previous_path)
    assert previous['all_checks_passed']
    for path, expected in previous['artifact_hashes'].items():
        assert digest(ROOT / path) == expected, path
    assert digest(ROOT / 'scripts/size_gate_research_verify.py') == previous['verifier_sha256']
    primary = ROOT / 'results/size_gate_study_20261009/final_checks.json'
    assert digest(primary) == previous['primary_verification_sha256']
    for path, expected in read(primary)['output_hashes'].items():
        assert digest(ROOT / path) == expected, path
    manifest = read(ROOT / 'data/size_gate_study/analysis_private/manifest.json')
    for path, expected in read(ROOT / 'data/size_gate_preflight/protected_file_hashes.json').items():
        assert digest(ROOT / path) == expected, path
    check_pins(manifest['protocol'])
    study = ROOT / 'results/size_gate_occlusion_diagnosis_20261009'
    summary, rows = read(study / 'summary.json'), read(study / 'cases.json')
    assert summary['cases'] == len(rows) == 80
    assert summary['script_sha256'] == digest(ROOT / 'scripts/size_gate_occlusion_diagnosis.py')
    assert summary['model_script_sha256'] == digest(ROOT / 'scripts/size_gate_viewpoint_diagnosis.py')
    assert summary['primary_summary_sha256'] == digest(primary.parent / 'summary.json')
    unique = set()
    for row in rows:
        key = (row['object'], row['angle_deg'], row['hidden_side'], row['requested_bbox_occlusion_fraction'])
        assert key not in unique
        unique.add(key)
        full, current = row['reference'], row['occluded']
        shift = np.asarray(current['centroid_camera_RDF_m']) - full['centroid_camera_RDF_m']
        np.testing.assert_allclose(shift, row['centroid_shift_RDF_m'], rtol=0, atol=1e-12)
        assert abs(np.linalg.norm(shift)-row['centroid_shift_norm_m']) < 1e-12
        assert abs(np.linalg.norm(shift[[0, 2]])-row['level_camera_horizontal_plane_shift_m']) < 1e-12
        assert abs(1-current['visible_pixels']/full['visible_pixels']-row['actual_mask_area_removed_fraction']) < 1e-12
        for alpha in [1., 2.]:
            assert row['gates'][str(alpha)] == gate_details(full['map_row'], current['map_row'], 1+alpha*(1.35-1), alpha*.1)
        if row['requested_bbox_occlusion_fraction'] == 0:
            assert np.all(shift == 0)
    for alpha in [1., 2.]:
        expected = dict(nonzero_cases=72,
                        rejected=sum(not r['gates'][str(alpha)]['admitted'] for r in rows if r['requested_bbox_occlusion_fraction'] > 0),
                        admitted_shift_over_0_1m=sum(r['gates'][str(alpha)]['admitted'] and r['centroid_shift_norm_m'] > .1 for r in rows))
        assert summary['counts'][str(alpha)] == expected
    plot = read(study / 'plot_provenance.json')
    for key, path in [('script_sha256', ROOT / 'scripts/size_gate_occlusion_plot.py'), ('summary_sha256', study / 'summary.json'),
                      ('cases_sha256', study / 'cases.json'), ('masks_sha256', study / 'display_masks.npz'),
                      ('figure_sha256', study / 'occlusion_centroid_readable.png'),
                      ('original_report_sha256', study / 'report_original.md'), ('current_report_sha256', study / 'report.md')]:
        assert digest(path) == plot[key]
    assert (study / 'report_original.md').read_text().replace('(occlusion_centroid.png)', '(occlusion_centroid_readable.png)') == (study / 'report.md').read_text()
    note = ROOT / 'notes/size_gate_occlusion_followup_20261009.md'
    for link in re.findall(r'\]\(([^)]+)\)', note.read_text()):
        assert (note.parent / link).resolve().is_file(), link
    for name in ['size_gate_occlusion_diagnosis', 'size_gate_occlusion_plot']:
        source = Path('/tmp') / f'{name}.log'
        assert source.is_file()
        target = study / f'{name}.log'
        if target.exists():
            assert digest(target) == digest(source)
        else:
            shutil.copyfile(source, target)
    out.mkdir()
    (out / 'index.md').write_text(
        '# Итог исследования размерной фильтрации TCAFF\n\n'
        '[Вывод по гипотезе](../../notes/size_gate_occlusion_followup_20261009.md). '
        '[Основной эксперимент](../size_gate_study_20261009/report.md).\n\n'
        '[Предыдущие диагностические проверки и команды](../size_gate_research_20261009/index.md). '
        '[Добавленная модель перекрытия](../size_gate_occlusion_diagnosis_20261009/report.md).\n\n'
        'Основной B0/ослабление: 420 прогонов, неизменённая полная разметка 798 пар. '
        '14 последующих диагностических проверок, включая пассивный аудит меток. '
        'Общее улучшение ослаблением не подтверждено; ракурс и перекрытие показаны как возможные механизмы в моделях. '
        'Их причинная роль в реальной записи пока не установлена.\n\n'
        'Все результаты одной записи зависимы. Другой записи нет. '
        '[Итоговая целостность](final_checks.json); предыдущие записи проверки не перезаписаны.\n')
    artifacts = [note, out / 'index.md', ROOT / 'scripts/size_gate_occlusion_diagnosis.py', ROOT / 'scripts/size_gate_occlusion_plot.py']
    artifacts.extend(p for p in study.rglob('*') if p.is_file())
    record = dict(all_checks_passed=True, utc_time=datetime.now(timezone.utc).isoformat(),
                  previous_verification_sha256=digest(previous_path), previous_artifacts_unchanged=True,
                  primary_outputs_unchanged=True, frozen_bundle_and_upstream_unchanged=True,
                  labels_sha256=previous['labels_sha256'], complete_annotation_count=798,
                  primary_verified_run_files=420, completed_followup_studies=14,
                  occlusion_model_cases=80, occlusion_cases_consistent=True,
                  plot_changes_layout_only=True,
                  artifact_hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted(artifacts)},
                  verifier_sha256=digest(Path(__file__)), command=sys.argv,
                  scope='Integrity and saved model consistency; not independent real-data validation')
    (out / 'final_checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ['all_checks_passed', 'previous_artifacts_unchanged', 'primary_outputs_unchanged',
                                           'complete_annotation_count', 'completed_followup_studies', 'occlusion_model_cases']}, indent=2))


if __name__ == '__main__':
    main()
