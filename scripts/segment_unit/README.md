# Аудит сегментации FastSAM и проверка рабочей гипотезы (отчёт, §8–9)

Два этапа: аудит на выборке из 800 кадров и проверка правил стабилизации единицы сегмента на всей записи. Протокол проверки зафиксирован до запуска в [configs/unit_hypothesis.json](../../configs/unit_hypothesis.json).

## Аудит на выборке (§8)

| Шаг | Скрипт | Окружение | Что делает |
|---|---|---|---|
| 1 | `extract_frames.py WORKSPACE OUT_DIR 200` | `.venv` | По 200 кадров на робота, равномерно по записи: цветной кадр L515, выровненная глубина с той же меткой времени, K, GT-поза |
| 2 | `segment.py WEIGHTS OUT_DIR OBJECTS_TXT` | `.venv-vision` | FastSAM-x с параметрами обёртки TCAFF, измерения TCAFF, статистика глубины в маске и вокруг неё, состав маски по GT-коробкам, сопоставление с сохранёнными детекциями авторов |
| 3 | `summarize.py OUT_DIR OUT_JSON` | `.venv` | Доли с 95% интервалами бутстрепа по кадрам → [sample_audit.json](../../results/segment_unit/sample_audit.json) |

Список кадров выборки — [sample_frames.json](../../results/segment_unit/sample_frames.json).

## Проверка гипотезы на всей записи (§9)

| Шаг | Скрипт | Окружение | Что делает |
|---|---|---|---|
| 1 | `run_all.sh WORKSPACE OUT_DIR WEIGHTS` | оба | Для каждого робота: `extract_all.py` (все кадры сохранённых детекций), `segment_all.py` (свойства всех масок без самих масок), удаление кадров |
| 2 | `build_variants.py OUT_DIR VARIANTS_DIR` | любой Python | Детекции вариантов R0, U1, U2, U3 в формате авторских JSON и метрики механизма |
| 3 | `run_variants.sh WORKSPACE VARIANTS_DIR RUNS_DIR 5 4` | `.venv` | Release-конфигурация TCAFF на каждом варианте, 5 повторов; меняется только `fastsam3d_data.root` |
| 4 | `evaluate_run.py` | `.venv` | Вызывается из шага 3: replay временного фильтра должен совпасть с прогоном, GT пересобирается независимо; метрики прогона и хэши входов |
| 5 | `aggregate.py RUNS_DIR MECHANISM_JSON OUT_JSON` | любой Python | Среднее и размах по повторам и правило решения протокола |
| 6 | `plot_unit_test.py` | `.venv` (matplotlib) | Рисунок §9 из сохранённой сводки → `results/figures/publication/unit_hypothesis.png`, хэши входа и выходов в `unit_hypothesis_provenance.json` |

**Окружения.** `.venv` — окружение бейслайна с robotdatapy, rosbags и clipperpy. `.venv-vision` — FastSAM (CASIA, коммит 4d153e9), ultralytics 8.0.120, PyTorch 2.4.1 CPU; версии записаны в протоколе.

**Веса** FastSAM-x скачиваются отдельно: `https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-x.pt`, SHA-256 указан в протоколе. В репозиторий веса не входят.

**Чтение глубины.** `compressedDepth` записан в формате RVL; после 20-байтового заголовка идёт поток RVL, декодер векторизован на NumPy (`rvl_decode` в `segment.py` и `segment_all.py`).

Скрипты сохранены в выполненном виде, с двумя оговорками. У `run_all.sh` пути рабочей области, вывода и весов вынесены в аргументы. В `segment_all.py` после обработки RR01 исправлено падение на кадре, где FastSAM не нашла ни одной маски (первый такой кадр — в RR04); на результаты остальных кадров исправление не влияет, обработка продолжена с места остановки.
