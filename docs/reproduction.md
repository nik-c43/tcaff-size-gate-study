# Воспроизведение и границы проверенных команд

Два пути разделены. A читает собственную разметку и сохранённые вычисленные ошибки, без авторского датасета. B запускает исходные компоненты с отдельно предоставленными данными. Никакой путь не требует ROMAN. Установка с нуля **не проверена**; проверки упаковки выполнены в существующем рабочем окружении.

## A. Проверка сохранённых результатов

Из корня, Python 3.12.3 (стандартная библиотека):

```bash
python3 scripts/inspect_saved.py
python3 scripts/read_annotations.py --task P0002
python3 scripts/read_annotations.py --html /tmp/tcaff-labels.html
```

Для сохранения нового протокола проверки:

```bash
python3 scripts/inspect_saved.py --output /tmp/tcaff-saved-checks.json
```

Выходной файл должен быть новым. Ожидаются 798 ответов, 420 файлов прогонов, 3598/4044 доступных оценок, 0.4379036843 м / 2.4461272324°, 1212/1740 и 1249/1740 правильных поздних обновлений, отношение времени 4.7340976. [Выполненная проверка](../results/final_saved_checks.json).

Проверка независимо агрегирует **сохранённые ошибки**, проверяет решения допуска по сохранённым размерам и целостность переносимых файлов. Она не повторяет вычисление позы, нейросети, replay фильтра или композицию GT. Результаты последних двух проверок читает из прежнего baseline audit. Их повтор требует внешних предложений и одометрии.

### Восстановление рисунков

Готовые PNG/SVG находятся в [results/figures/publication](../results/figures/publication). В существующем окружении проверена команда:

```bash
export MPLCONFIGDIR=/tmp/tcaff-study-mpl
.venv/bin/python scripts/render_saved.py --output /tmp/tcaff-figures
```

Для отдельного окружения только рисунков предлагается, но с нуля не проверено:

```bash
python3 -m venv .venv-plots
.venv-plots/bin/python -m pip install -r requirements-light.txt
MPLCONFIGDIR=/tmp/tcaff-study-mpl .venv-plots/bin/python scripts/render_saved.py \
  --output /tmp/tcaff-figures-clean
```

Рисунки строятся по `summary.json`, `cases.json` и **собственным идеальным** `display_masks.npz`; авторские изображения не нужны. Новый `plot_provenance.json` фиксирует SHA входов, кода и выходов. Это изменение оформления, не повтор эксперимента. Старые рисунки и их provenance сохранены отдельно.

### Повторная проверка существующей модели

После подключения submodules и зависимостей пути B, без авторских данных:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/reproduce_model.py --output /tmp/tcaff-model-checks.json
```

Команда проверена в рабочем окружении: 80 случаев, 8 контролей, максимальное расхождение с сохранёнными числами 0.0. Геометрические функции совпадают по AST с выполненным архивом. Вызывается исходный размерный допуск менеджера; решатель CLIPPER и временной фильтр не запускаются. [Протокол](../results/model_checks.json).

## B. Внешние данные и повторный запуск

### Зависимости и закрепление

При появлении опубликованного URL клонировать с submodules:

```bash
git clone --recurse-submodules "$STUDY_REPOSITORY_URL" tcaff-size-gate-study
cd tcaff-size-gate-study
git submodule update --init --recursive
git -C vendor/tcaff rev-parse HEAD
git -C vendor/clipper rev-parse HEAD
```

URL ещё не создан; это инструкция после публикации, не выполненное сетевое клонирование. Ожидаемые коммиты:

| Зависимость | Коммит |
|---|---|
| TCAFF | `ff4ceab04b03fecabc4de9c9cbe7dbba4ab50df9` |
| CLIPPER | `e514dc29c273837ffdfeebbefbdcb2a93d970969` |
| robotdatapy | `0e7853d63b424ea4b074eb6e203d82ad5a9ecf5c` |
| plot_utils | `fab133e1c187174062abd2799d727a75cb826021` |

[requirements.txt](../requirements.txt) — snapshot рабочего CPU-окружения, с удалённой только локальной `file://` ссылкой на clipperpy; он собирается отдельно. TCAFF `setup.py` целиком не устанавливался: это избежало установки ненужного нейросетевого frontend. Не требуется download моделей. Источник и build-зависимости — [provenance](provenance.md).

Команды установки взяты из исходного рабочего проекта и адаптированы **только по расположению CLIPPER**. Они документированы, но новая чистая установка в этом запросе не выполнена:

```bash
sudo apt-get install build-essential cmake python3-dev python3-venv \
  libeigen3-dev libblas-dev liblapack-dev libgtest-dev
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cmake -S vendor/clipper -B vendor/clipper/build \
  -DCLIPPER_BUILD_TESTS=ON -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build vendor/clipper/build -j2
vendor/clipper/build/test/tests
.venv/bin/python -m pip install --no-build-isolation --no-deps \
  ./vendor/clipper/build/bindings/python
.venv/bin/python -m pip check
```

Eigen/GTest — системные development-пакеты. В закреплённом CLIPPER используется `CLIPPER_BUILD_TESTS`, не старое имя `BUILD_TESTS`. CMake FetchContent требует сеть для закреплённых pybind11/PMC/SCS; автоматической загрузки исследовательского датасета здесь нет. При отличающейся версии Python/компилятора сборка и бинарный SHA могут отличаться. Не подменять это молчаливым обновлением алгоритма.

### Данные и внешняя рабочая область

Получить авторские данные отдельно по [инструкции](../data/README.md). Задать переменные своими **внешними** путями:

```bash
export TCAFF_DATA_DIR=/path/to/external/tcaff_mot_data
export STUDY_VENV_DIR=/path/to/prepared/venv
export STUDY_WORK_DIR=/path/to/new/external/workspace
```

Это placeholders, не готовые пути. `STUDY_WORK_DIR` должен отсутствовать и лежать вне репозитория. Подготовка сохраняет старые файлы и переносит неизменённые runners из `scripts/original` в ожидаемую ими структуру. Входные данные подключаются ссылками, приватный пакет копируется в отдельную рабочую область, чтобы не писать в исходник.

### Новый release-baseline

```bash
python3 scripts/prepare_workspace.py --workspace "$STUDY_WORK_DIR" \
  --dataset "$TCAFF_DATA_DIR" --venv "$STUDY_VENV_DIR"
cd "$STUDY_WORK_DIR"
bash scripts/run_tcaff_baseline.sh \
  results/tcaff_baseline_release data/tcaff_baseline_release
```

Исходная проверенная команда запускалась в старом проекте с этими относительными каталогами. Перенос runner, конфигурации и проверка входов выполнены; **полный baseline из новой рабочей области не перезапускался**, поскольку экспериментальный код не менялся и исходный replay уже проверен. Случайный CLIPPER не обязан воспроизвести точные прежние числа нового полного прогона. Внешние результаты нельзя автоматически включать в репозиторий: захваты карт содержат авторскую геометрию.

### Точный прежний размерный эксперимент

Одного скачанного датасета недостаточно для побитового повторения прежнего выбора: нужны внешние замороженные карты/предложения, приватные `manifest.json`, `tasks.json`, `bundle_frozen.json`, `label_counts.json` и baseline audit. Их SHA зафиксированы в [inventory](source_inventory.json), [selection](../results/size_gate/selection.json) и результатах. Снимок находится в исходном рабочем проекте, **не распространяется здесь вместе с датасетом**.

```bash
export STUDY_FROZEN_PROJECT=/path/to/external/original/research-snapshot
export STUDY_WORK_DIR=/path/to/new/external/frozen-workspace
python3 scripts/prepare_workspace.py --workspace "$STUDY_WORK_DIR" \
  --dataset "$TCAFF_DATA_DIR" --venv "$STUDY_VENV_DIR" \
  --frozen-project "$STUDY_FROZEN_PROJECT"
cd "$STUDY_WORK_DIR"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/tcaff-study-mpl .venv/bin/python scripts/size_gate_evaluate.py count \
  --labels data/size_gate_study/labels.json --output results/check_label_counts.json
bash scripts/run_size_gate_study.sh \
  data/size_gate_study/labels.json data/size_gate_runs_reproduction
```

Подготовка и **подсчёт count проверены** в новой внешней рабочей области. Последняя команда — проверенный исторический runner; новую серию 420 прогонов в рамках подготовки не запускали. Runner сначала проверяет полную разметку/хэши, затем development, замораживает выбор и запускает holdout. Выходной каталог должен быть новым. Существующие результаты не переписываются.

В режиме frozen проверяется SHA исходного clipperpy-бинарника. Его путь разрешено переместить, **сам бинарник должен совпадать**. Новая сборка может не пройти этот строгий guard; нужно либо исходное окружение snapshot, либо отдельно оформить новый baseline и новый протокол, не выдавая его за точное повторение. Обход guard или изменение исходных ответов здесь не предусмотрены.

Если прежнего приватного пакета нет, исходные `size_gate_prepare.py`, `size_gate_images.py`, `size_gate_finalize.py` сохраняют процедуру формирования. Однако пересоздание карт/пакета может изменить fingerprint и отбор; исходная разметка не назначается новому пулу автоматически. Это ограничение повторного тяжёлого запуска, а не препятствие чтению и проверке всех готовых ответов по пути A.

### Что остаётся внешним или непроверенным

- Просмотр реальных RGB/глубин, повторный GT/replay и новые предложения MNO требуют внешних данных.
- Полный нейросетевой frontend и MOT/MOTA не проверены, модели не установлены этим репозиторием.
- Чистая установка, новое сетевое клонирование и повтор всех 420 прогонов из опубликованной копии не проверены.
- ROMAN отсутствует в обеих ветках. Исторические скрипты с его импортами сохранены только в исходном проекте.

[Проверки, реально выполненные при подготовке](validation.md), [происхождение](provenance.md).
