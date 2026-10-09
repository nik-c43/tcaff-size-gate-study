# Внешние авторские данные

В этом каталоге нет датасета. Авторский TCAFF MOT dataset нужно получить отдельно: [инструкция upstream](https://github.com/mit-acl/tcaff/tree/ff4ceab04b03fecabc4de9c9cbe7dbba4ab50df9#dataset) направляет на [авторскую форму доступа](https://forms.gle/SnWwHHnF6gPugVtK7). Форма и доступ не проверялись отправкой; запрос от имени пользователя не выполнялся.

Ожидаемая внешняя структура для component baseline:

```text
tcaff_mot_data/
  fastsam_data/{RR01,RR04,RR06,RR08}.json
  calib.yaml
  calib_xyzquat.yaml
  objects.txt
  data/kimera_odom/{RR01,RR04,RR06,RR08}.bag
  data/RR04_compressed-001.bag
  data/RR06_compressed-002.bag
  data/RR01_compressed-003.bag
  data/RR08_compressed-004.bag
```

Данные описывают одну запись четырёх роботов с шестью пешеходами и статическими коробками. RGB/depth/IMU RealSense D455/L515 и mocap доступны в bags; baseline по готовым JSON не запускает FastSAM заново. Изображения в ручной разметке восстанавливаются из bags; исходные маски/классы сегментов в предоставленных JSON отсутствуют.

Путь задаётся через `--dataset` при [подготовке внешнего workspace](../docs/reproduction.md); ничего не скачивается при установке или лёгкой проверке. Bags, полные JSON, изображения, архивы и локальный пул разметки остаются вне публикуемого репозитория.

Для точного прежнего размерного эксперимента дополнительно нужен внешний исследовательский snapshot с замороженными картами, приватным manifest и audit. Публичные метки и ссылки на их происхождение находятся в [annotations](../annotations/README.md), все 798 ответов доступны без датасета. Условия распространения авторских изображений не установлены; готовые реальные иллюстрации сюда не включены.
