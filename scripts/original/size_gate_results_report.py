"""Summarize completed frozen runs and audit only their selected global gate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from size_gate_evaluate import validate_labels
from size_gate_prepare import check_pins
from size_gate_label_audit import support
from h1_map_audit import object_map
from tcaff.tcaff.tcaff_manager import TCAFFManager


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return f'{100*value:.2f}%'


def number(value):
    return '—' if value is None else f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='data/size_gate_runs_20261009')
    parser.add_argument('--output', default='results/size_gate_study_20261009')
    args = parser.parse_args()
    runs, out = ROOT / args.runs, ROOT / args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT / 'results'):
        raise ValueError('Use a new separate results directory')
    study = ROOT / 'data/size_gate_study'
    manifest = read(study / 'analysis_private/manifest.json')
    frozen = read(study / 'bundle_frozen.json')
    assert digest(study / 'analysis_private/manifest.json') == frozen['manifest_sha256']
    for filename, expected in manifest['provenance']['code_hashes'].items():
        assert digest(ROOT / filename) == expected, filename
    check_pins(manifest['protocol'])
    labels_path = runs / 'labels_snapshot.json'
    labels = validate_labels(manifest, read(labels_path))
    note_flags = {name: [task_id for task_id, answer in labels.items()
                         if token in answer.get('notes', '').casefold()]
                  for name, token in [('robot_mentioned', 'робот'), ('person_mentioned', 'человек')]}
    label_hash = digest(labels_path)
    counts = read(study / 'analysis_private/label_counts.json')
    development = read(runs / 'development/summary.json')
    selection = read(runs / 'development/selection.json')
    cold = read(runs / 'holdout/summary.json')
    full = read(runs / 'holdout/full_history_summary.json')
    for record in [counts, development, selection, cold]:
        assert record['labels_sha256'] == label_hash
    assert selection['development_summary_sha256'] == digest(runs / 'development/summary.json')
    alpha = selection['alpha']
    assert set(cold['summaries']) == {'1.0', str(alpha)}
    original, baseline, relaxed = full['frozen_B0_holdout_context'], full['rerun_holdout_context']['1.0'], full['rerun_holdout_context'][str(alpha)]
    pairs = sorted({e['pair'] for e in manifest['episodes']})
    source = {pair: read(ROOT / manifest['protocol']['baseline'] / (pair + '.json')) for pair in pairs}
    before_params = source[pairs[0]]['params']
    ratio = 1 + alpha * (before_params['wh_scale_diff'] - 1)
    difference = alpha * before_params['h_diff']
    selected_table = {label: Counter() for label in ['same_object', 'different_objects', 'part_whole_or_ambiguous', 'cannot_tell']}
    exclusion_reasons = {label: Counter() for label in selected_table}
    failed_checks = {label: Counter() for label in selected_table}
    selected_high = {label: Counter() for label in selected_table}
    for task in manifest['tasks']:
        label = labels[task['task_id']]['label']
        if not task['gate']['admitted']:
            exclusion_reasons[label][task['gate']['first_exclusion_reason']] += 1
            failed_checks[label].update(task['gate']['failed_checks'])
    annotation_support, candidate_counts = [], {'1.0': [], str(alpha): []}
    update_runtimes = {'1.0': [], str(alpha): []}
    common_rows, verified_runs = [], 0
    for stage, expected_runs in [('development', 240), ('holdout', 120)]:
        paths = list((runs / stage).glob('E*_a*_r*.json'))
        assert len(paths) == expected_runs
        for path in paths:
            run = read(path)
            assert run['temporal_replay_verified']
            original_kwargs = {k: v for k, v in source[run['pair']]['params'].items() if k != 'ts'}
            expected = dict(original_kwargs,
                            wh_scale_diff=1+run['alpha']*(original_kwargs['wh_scale_diff']-1),
                            h_diff=run['alpha']*original_kwargs['h_diff'])
            assert run['params'] == expected
            verified_runs += 1
    full_cache = {}
    for pair in pairs:
        base_kwargs = {k: v for k, v in source[pair]['params'].items() if k != 'ts'}
        for repeat in range(1, 6):
            for a in [1., alpha]:
                run = read(runs / 'holdout' / f'full_{pair}_a{a}_r{repeat}.json')
                expected = dict(base_kwargs, wh_scale_diff=1+a*(base_kwargs['wh_scale_diff']-1), h_diff=a*base_kwargs['h_diff'])
                assert run['params'] == expected and run['temporal_replay_verified']
                assert len(run['frames']) == len(source[pair]['frames']) == 337
                full_cache[pair, a, repeat] = run
                verified_runs += 1
    assert verified_runs == 420
    for episode in manifest['episodes']:
        pair, step = episode['pair'], episode['anchor']
        frame = source[pair]['frames'][step]
        ma, mb = np.asarray(frame['map_a']).reshape(-1, 6), np.asarray(frame['map_b']).reshape(-1, 6)
        params = source[pair]['params']
        kwargs = {k: v for k, v in params.items() if k != 'ts'}
        kwargs.update(wh_scale_diff=1+alpha*(params['wh_scale_diff']-1), h_diff=alpha*params['h_diff'])
        manager = TCAFFManager(**kwargs)
        allowed = manager.get_putative_assoc(object_map(ma), object_map(mb))
        allowed_set = set(map(tuple, allowed.tolist()))
        assert len(allowed) == full_cache[pair, alpha, 1]['frames'][step]['putative_size_admitted_pairs']
        tasks = [t for t in manifest['tasks'] if t['episode_id'] == episode['episode_id']]
        high_selected = []
        for task in tasks:
            edge = task['left']['map_index'], task['right']['map_index']
            admitted = edge in allowed_set
            label = labels[task['task_id']]['label']
            selected_table[label]['admitted' if admitted else 'rejected'] += 1
            if labels[task['task_id']]['confidence'] == 'high':
                selected_high[label]['admitted' if admitted else 'rejected'] += 1
            if admitted and labels[task['task_id']]['label'] == 'same_object' and labels[task['task_id']]['confidence'] == 'high':
                high_selected.append(task)
        annotation_support.append(dict(episode_id=episode['episode_id'], split=episode['split'],
                                       selected_high_confidence=support(high_selected, ma, mb, params)))
        if episode['split'] != 'holdout':
            continue
        for repeat in range(1, 6):
            b = full_cache[pair, 1., repeat]['frames'][episode['start']:episode['end']+1]
            r = full_cache[pair, alpha, repeat]['frames'][episode['start']:episode['end']+1]
            assert [f['time'] for f in b] == [f['time'] for f in r]
            common = [(fb, fr) for fb, fr in zip(b, r) if fb['error'] is not None and fr['error'] is not None]
            common_rows.append(dict(episode_id=episode['episode_id'], repeat=repeat, updates=len(common),
                                    baseline_translation_m=None if not common else float(np.mean([fb['error'][0] for fb, fr in common])),
                                    relaxed_translation_m=None if not common else float(np.mean([fr['error'][0] for fb, fr in common])),
                                    baseline_yaw_deg=None if not common else float(np.mean([fb['error'][1] for fb, fr in common])),
                                    relaxed_yaw_deg=None if not common else float(np.mean([fr['error'][1] for fb, fr in common]))))
            for a, frames in [('1.0', b), (str(alpha), r)]:
                candidate_counts[a].append(float(np.mean([f['putative_size_admitted_pairs'] for f in frames])))
                update_runtimes[a].extend(f['runtime_s'] for f in frames)
    common_episode = []
    for eid in sorted({row['episode_id'] for row in common_rows}):
        rows = [row for row in common_rows if row['episode_id'] == eid]
        item = dict(episode_id=eid, mean_common_updates=float(np.mean([r['updates'] for r in rows])))
        for key in ['baseline_translation_m', 'relaxed_translation_m', 'baseline_yaw_deg', 'relaxed_yaw_deg']:
            values = [r[key] for r in rows if r[key] is not None]
            item[key] = None if not values else float(np.mean(values))
        common_episode.append(item)
    common_summary = {}
    for key in ['baseline_translation_m', 'relaxed_translation_m', 'baseline_yaw_deg', 'relaxed_yaw_deg']:
        values = [row[key] for row in common_episode if row[key] is not None]
        common_summary[key] = None if not values else float(np.mean(values))
    summary = dict(alpha=alpha, selected_wh_scale_diff=ratio, selected_h_diff_m=difference,
                   labels_sha256=label_hash, wrong_first_constraint_satisfied=selection['wrong_first_constraint_satisfied'],
                   frozen_B0=original, rerun_B0=baseline, selected_variant=relaxed,
                   cold_start=cold['summaries'], selected_gate_label_counts={k: dict(v) for k, v in selected_table.items()},
                   raw_B0_label_counts=counts['pair_observation_counts'],
                   raw_B0_label_counts_by_confidence=counts['by_confidence'],
                   explicit_note_flags={name: len(ids) for name, ids in note_flags.items()},
                   selected_high_confidence_label_counts={k: dict(v) for k, v in selected_high.items()},
                   B0_first_exclusion_reasons_by_label={k: dict(v) for k, v in exclusion_reasons.items()},
                   B0_all_failed_checks_by_label={k: dict(v) for k, v in failed_checks.items()},
                   selected_high_confidence_five_clique_episodes=sum(row['selected_high_confidence']['clique_at_least_5'] for row in annotation_support),
                   common_accepted_updates=common_summary,
                   mean_candidates_per_holdout_update={a: float(np.mean(v)) for a, v in candidate_counts.items()},
                   engine_update_latency={a: dict(updates=len(v), mean_s=float(np.mean(v)),
                                                 p95_s=float(np.percentile(v,95)), max_s=float(np.max(v)),
                                                 updates_over_1s=sum(t>1. for t in v)) for a,v in update_runtimes.items()},
                   verified_run_files=verified_runs, temporal_replay_verified_for_all_runs=True,
                   interpretation='Primary full-history holdout comparison; episode means of five repeats from one correlated recording. Cold starts are supplemental. Common-support repeats are paired by repeat index, not shared random seeds.',
                   manual_semantics='Recorded identity labels may mix boxes and groups; semantic distinctness does not imply geometric uselessness in XY.',
                   provenance={str(p.relative_to(ROOT)): digest(p) for p in [labels_path, runs/'run_info.json',
                               runs/'development/summary.json', runs/'development/selection.json',
                               runs/'holdout/summary.json', runs/'holdout/full_history_summary.json',
                               runs/'environment.json',
                               study/'analysis_private/label_counts.json', ROOT/'results/size_gate_full_annotation_audit/summary.json']},
                   report_script_sha256=digest(Path(__file__)), command=sys.argv)
    check_pins(manifest['protocol'])
    assert digest(labels_path) == label_hash
    out.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (out / 'analysis_private').mkdir()
    (out / 'analysis_private/common_support.json').write_text(json.dumps(common_rows, indent=2) + '\n')
    (out / 'analysis_private/selected_gate_support.json').write_text(json.dumps(annotation_support, indent=2) + '\n')
    (out / 'analysis_private/explicit_note_flags.json').write_text(json.dumps(note_flags, indent=2) + '\n')
    names = ['Сохранённый B0', 'Повторы B0', f'Ослабление α={alpha}']
    metrics = [('availability', 'Доступность, %', 100), ('correct_update_availability', 'Правильные оценки, %', 100),
               ('mean_translation_m', 'Ошибка переноса, м', 1), ('mean_yaw_deg', 'Ошибка угла, °', 1)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (key, title, scale) in zip(axes.flat, metrics):
        vals = [None if record[key] is None else record[key] * scale for record in [original, baseline, relaxed]]
        plotted = [0. if v is None else v for v in vals]
        bars = ax.bar(names, plotted, color=['#8899aa', '#355f99', '#df8637'])
        ax.bar_label(bars, labels=['нет оценок' if v is None else f'{v:.2f}' for v in vals], padding=4, fontsize=10)
        ax.set_title(title)
        ax.set_ylim(0, max(plotted) * 1.18 if max(plotted) else 1)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Размерный фильтр TCAFF: отложенные окна с полной историей', fontsize=15)
    fig.text(.5, .015, 'Одна запись; 12 зависимых окон; пять повторов. Ошибки условны на наличии оценки.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .045, 1, .94])
    fig.savefig(out / 'holdout_comparison.png', dpi=160)
    plt.close(fig)
    lines = ['# Проверка размерного фильтра TCAFF — 9 октября 2026', '',
             'Проверяемая гипотеза: исходный размерный допуск может устранять полезную геометрическую поддержку; '
             'одно глобальное ослабление может восстановить её и улучшить конечное совмещение. '
             'Потеря поддержки и улучшение совмещения оцениваются отдельно. '
             'Причина изменения размеров — например, частичное перекрытие — в этой проверке не устанавливается.', '',
             f'Полная разметка: 798 ответов. Исходный экспорт `data/labels(5).json` сохранён; '
             f'рабочая копия `data/size_gate_study/labels.json`. SHA256: `{label_hash}`.', '',
             f'На development выбран **α={alpha}**: отношение размеров ≤{ratio:g}, абсолютная разность ≤{difference:g} м. '
             f'Исходные ограничения: ≤1.35 и ≤0.1 м. Ограничение по неверным первым решениям на development: ' +
             ('выполнено.' if selection['wrong_first_constraint_satisfied'] else '**нарушено; выбор минимального α по правилу fallback не считается успехом.**'), '',
             'Остальные параметры, карты, стоимость q², временной фильтр и upstream сохранены. '
             'Основная проверка использует полную историю от начала записи. Холодные старты в окнах по 29 обновлений приведены отдельно.', '',
             'Вход алгоритма: неизменённые карты объектов и GT в `data/tcaff_baseline_release/*.json`, '
             '337 обновлений на поток. Их исходные наблюдения связаны с `data/tcaff_mot_data/fastsam_data/*.json` '
             'в закрытом manifest пакета. Точность RGB-разметки ограничена отсутствием масок. '
             'В алгоритмических прогонах GT используется только для оценки результата.', '',
             '## Основное сравнение на отложенных окнах', '',
             '| Метрика | Сохранённый B0 | Повторы B0 | Выбранное ослабление |', '|---|---:|---:|---:|']
    for key, title, scale in metrics:
        lines.append('| ' + title + ' | ' + ' | '.join(number(None if record[key] is None else record[key]*scale) for record in [original, baseline, relaxed]) + ' |')
    lines += [f"| Время алгоритма на окно, с | — | {baseline['runtime_s']:.4f} | {relaxed['runtime_s']:.4f} |", '',
              'Правильная оценка: ошибка ≤1 м и ≤5°. Ошибки усреднены сначала по повторам внутри эпизода, затем по эпизодам; '
              'это не единая средняя по всем кадрам. Доступность включает и ошибочные оценки. '
              'Сохранённый B0 не перезаписан; новые повторы показывают вариацию стохастического CLIPPER.', '',
              f"Среднее число всех кандидатов до CLIPPER на отложенном обновлении: {summary['mean_candidates_per_holdout_update']['1.0']:.2f} → "
              f"{summary['mean_candidates_per_holdout_update'][str(alpha)]:.2f}. Время измерено на той же машине с одним потоком; "
              'версии и CPU сохранены в `environment.json` каталога прогонов.', '',
              f"95-й процентиль времени обновления: {summary['engine_update_latency']['1.0']['p95_s']:.4f} → "
              f"{summary['engine_update_latency'][str(alpha)]['p95_s']:.4f} с; обновлений дольше 1 с: "
              f"{summary['engine_update_latency']['1.0']['updates_over_1s']} → {summary['engine_update_latency'][str(alpha)]['updates_over_1s']}. "
              'Это описание офлайн-вычислений на данной машине; онлайн-дедлайны и весь frontend не проверялись.', '',
              '![Сравнение на отложенных окнах](holdout_comparison.png)', '',
              '## Различия между отложенными эпизодами', '',
              '| Эпизод | Доступность B0 → вариант, % | Правильные обновления B0 → вариант, % | Перенос B0 → вариант, м | Угол B0 → вариант, ° |',
              '|---|---:|---:|---:|---:|']
    selected_episodes = {row['episode_id']: row for row in relaxed['per_episode']}
    for row in sorted(baseline['per_episode'], key=lambda row: row['episode_id']):
        changed = selected_episodes[row['episode_id']]
        entries = [row['episode_id']]
        for key, scale in [('availability',100), ('correct_update_availability',100),
                           ('mean_translation_m',1), ('mean_yaw_deg',1)]:
            entries.append(number(None if row[key] is None else row[key]*scale)+' → '+
                           number(None if changed[key] is None else changed[key]*scale))
        lines.append('| '+' | '.join(entries)+' |')
    lines += ['',
              'Каждая строка — среднее пяти повторов, а не отдельная независимая запись. '
              'Диапазоны каждого показателя между повторами сохранены в `summary.json` в `per_episode`. '
              'Прочерк означает отсутствие принятой оценки; такие окна остаются в доступности и доле правильных обновлений.', '',
              '## Дополнительная диагностика инициализации', '',
              '| Метрика холодного старта | B0 | Выбранное ослабление |', '|---|---:|---:|']
    b, r = cold['summaries']['1.0'], cold['summaries'][str(alpha)]
    for key, title in [('wrong_first_rate', 'Неверное первое решение'), ('no_first_rate', 'Нет первого решения'),
                       ('correct_by_horizon_rate', 'Правильная инициализация к концу окна')]:
        lines.append(f'| {title} | {percent(b[key])} | {percent(r[key])} |')
    lines += [f"| Время до правильной оценки с ограничением горизонтом, с | {number(b['capped_time_to_correct_s'])} | {number(r['capped_time_to_correct_s'])} |", '',
              'Отсутствие правильной оценки цензурировано концом окна, а не удалено из среднего. '
              'Эта диагностика сбрасывает временной фильтр и не подменяет B0 с полной историей. '
              'Первая глобальная инициализация всей записи находится в ранней части и не считается отложенным тестом.', '',
              '## Ручные категории и согласованность', '',
              '| Исходная ручная метка | Допущено B0 | Исключено B0 | Допущено выбранным вариантом | Исключено выбранным вариантом |', '|---|---:|---:|---:|---:|']
    for label in selected_table:
        lines.append(f"| `{label}` | {counts['pair_observation_counts'].get(label+'_admitted',0)} | {counts['pair_observation_counts'].get(label+'_rejected',0)} | {selected_table[label]['admitted']} | {selected_table[label]['rejected']} |")
    lines += ['',
              'По записанным категориям B0 исключает 260/314 (82.80%) пар «один объект» и 304/341 (89.15%) пар «разные объекты». '
              'Две неоднозначные категории не отнесены ни к правильным, ни к неправильным соответствиям. '
              'Все 798 пар треков различны; это не 798 независимых физических предметов или экспериментов.', '',
              '| Только высокая уверенность | Допущено B0 | Исключено B0 | Допущено выбранным вариантом | Исключено выбранным вариантом |',
              '|---|---:|---:|---:|---:|']
    for label in selected_table:
        lines.append(f"| `{label}` | {counts['by_confidence']['high'].get(label+'_admitted',0)} | {counts['by_confidence']['high'].get(label+'_rejected',0)} | {selected_high[label]['admitted']} | {selected_high[label]['rejected']} |")
    reason_names = [('width_ratio', 'Отношение ширин'), ('height_ratio', 'Отношение высот'),
                    ('width_difference', 'Разность ширин'), ('height_difference', 'Разность высот')]
    lines += ['', '| Первое сработавшее условие B0 | «Один объект» | «Разные объекты» |', '|---|---:|---:|']
    for key, title in reason_names:
        lines.append(f"| {title} | {exclusion_reasons['same_object'][key]} | {exclusion_reasons['different_objects'][key]} |")
    lines += ['', 'Первое условие зависит от порядка `if/elif` upstream; одна пара может нарушать несколько ограничений. '
              'Все нарушения сохранены отдельно в JSON и не суммируются как независимые исключения.']
    audit = read(ROOT / 'results/size_gate_full_annotation_audit/summary.json')
    high = audit['support']['same_object_high_only']
    lines += ['', f"При высокой уверенности до размерного фильтра набор из пяти взаимно согласованных пар `same_object` существует в {high['before_clique_at_least_5']} из 24 эпизодов; "
              f"после B0 — в {high['after_clique_at_least_5']}; после выбранного варианта — в {summary['selected_high_confidence_five_clique_episodes']}. "
              'Графы проверены по исходным ограничениям CLIPPER. Наличие такого набора не гарантирует, что решатель выберет его или временной фильтр примет правильную позу.', '',
              'Метки интерпретируются как записанные категории разметчика. Коробки и группы коробок могут задавать разные единицы; '
              'стол и светильник на одной вертикали могут быть разными предметами, но геометрически совместимыми в XY. '
              'Низкая уверенность не исправлялась автоматически. Маски отсутствуют. См. `notes/size_gate_annotation_caveats.md`.', '',
              f"В {len(note_flags['robot_mentioned'])} парах есть явный комментарий о роботах, в {len(note_flags['person_mentioned'])} — о человеке. "
              'Это неполная отметка содержимого изображений, а не классификация всех ориентиров. '
              'Одинаковый внешний вид роботов не устанавливает физическую идентичность; даже идентичность движущегося объекта '
              'не доказывает пригодность как статического ориентира между разновременными наблюдениями. '
              'Эти пары сохранены в эксперименте без автоматического исправления меток.', '',
              'Ручной пул ограничен 798 парами с 3D-близостью центров <0.6 м по GT на выбранных якорях. '
              'Ослабление меняет допуск всех пар объектов карты, включая пары за пределами этого пула. '
              'По ручным ответам нельзя оценить общее число новых ложных кандидатов; их совокупное влияние проверяется метриками позы.', '',
              '## Проверка на общих принятых обновлениях', '',
              f"На обновлениях, где обе конфигурации дали оценку: ошибка переноса {number(common_summary['baseline_translation_m'])} → {number(common_summary['relaxed_translation_m'])} м; "
              f"угла {number(common_summary['baseline_yaw_deg'])} → {number(common_summary['relaxed_yaw_deg'])}°. "
              'Сначала усреднение по повтору внутри эпизода, затем по эпизодам с общими принятыми обновлениями. '
              'Номер повтора сопоставлен; общие случайные начальные значения CLIPPER не задавались.', '',
              '## Воспроизводимость и ограничения', '',
              'Одна запись четырёх роботов. Шесть канонических потоков, 12 направленных пар, окна и повторы не являются независимыми экспериментами. '
              'Отложенная часть ранее просматривалась в аудите, но не использовалась для нового выбора α. '
              'Выборка специально обогащена потерей поддержки; переносимость на другие сцены не проверена. '
              'Полный нейросетевой frontend и MOT/MOTA не воспроизводятся. Слияния объектов, исправления стоимости и адаптивного ослабления нет.', '',
              f"TCAFF: `{manifest['protocol']['tcaff_commit']}`; CLIPPER: `{manifest['protocol']['clipper_commit']}`. "
              'Размерный допуск реализован в `TCAFFManager.get_putative_assoc` '
              '(`vendor/tcaff/tcaff/tcaff/tcaff_manager.py`); передача только XY-центров — в `get_frame_align_measurements`. '
              'Эксперимент меняет только `wh_scale_diff` и `h_diff` при создании фабрики. '
              'Число требуемых объектов, ограничения CLIPPER, параметры временного фильтра и стоимость неизменны.', '',
              f'Проверены {verified_runs} файлов прогонов: параметры отличаются только двумя размерными ограничениями; '
              'для каждого сохранены предложения, оценки, GT, ошибки и время, а точное воспроизведение временного фильтра прошло. '
              'Сохранены SHA256 разметки, выбора α, summary и кода. Сырые результаты: `' + args.runs + '`.', '',
              'Команды:', '', '```bash',
              'bash scripts/run_size_gate_study.sh data/size_gate_study/labels.json data/size_gate_runs_20261009',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_label_audit.py',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_results_report.py',
              'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/tcaff-mpl \\',
              '  .venv/bin/python scripts/size_gate_results_verify.py', '```', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'alpha': alpha, 'wrong_first_constraint_satisfied': selection['wrong_first_constraint_satisfied'],
                      'report': str(out/'report.md'), 'verified_runs': verified_runs,
                      'primary_B0': {key: baseline[key] for key, title, scale in metrics},
                      'primary_relaxed': {key: relaxed[key] for key, title, scale in metrics}}, indent=2))


if __name__ == '__main__':
    main()
