# Lidar Archaeology Segmentation (Hydra Prototype)

Этот прототип показывает, как можно вынести весь хардкод из `training.py` и `evaluation.py` в переиспользуемую архитектуру с помощью конфигурационных файлов **Hydra**. 

## Структура папок

- `configs/`: Содержит все YAML конфигурации. Разбито на модули (`dataset`, `model`, `training`, `paths`), чтобы можно было легко менять любую часть без правки логики. Файл `train.yaml` является главной точкой входа.
- `src/`: Основной код Python
  - `datasets/` & `models/`: Перенесённые без изменений ваши датасеты и модели, но теперь они инициализируются динамически через Hydra (см. `_target_` в конфигах).
  - `engine/`: Включает в себя `losses.py`, `metrics.py` и `trainer.py`. Класс `Trainer` теперь универсальный и не зависит от жестко прописанного датасета.
  - `train.py`: Скрипт полного цикла обучения.
  - `inference.py`: Специализированный скрипт для обработки огромных массивов данных (например, тайлов 35000x23000) без Ground Truth масок.

## Как проводить эксперименты

Hydra позволяет подменять и изменять параметры прямо из командной строки.

### Базовый запуск (Использует `configs/train.yaml`)
```bash
python src/train.py
```

### Замена датасета (Например, с `DEM` на `RGB`)
Вместо того чтобы лезть в код и переписывать пути:
```bash
python src/train.py dataset=rgb
```

### Настройка гиперпараметров "на лету"
Вы хотите поменять Loss или Learning Rate? Нет проблем:
```bash
python src/train.py training.learning_rate=0.0005 model.backbone=resnet50 training.epochs=5
```

### Включение WandB (Weights & Biases)
В базовом варианте логгер выключен (выводит просто в терминал). Включить его:
```bash
python src/train.py logger.use_wandb=true logger.name="experiment_resnet50_dem"
```

## Как производить огромный Inference?

Скрипт `inference.py` предназначен для предсказания всей площадки 35000х23000 в огромную бинарную карту.
Так как мы не имеем `Ground Truth`, флаг `dataset.no_gt=True` генерирует пустую маску-заглушку в памяти, чтобы ваш `Dataset` не ругался на отсутствие `mask_path`.

```bash
python src/inference.py checkpoint_path=/path/to/best_model.pth dataset=dem batch_size=16
```
> Результат сохранится в `outputs/full_prediction_map.npy` 

## Деплой на DGX через SLURM

Обратите внимание на файл `run_slurm.sh`. 
В нем мы задаем переменную `export DATA_ROOT=...`. В конфигах (в папке `configs/paths`) написано: `data_dir: ${oc.env:DATA_ROOT}`. Это значит, что код сам прочитает этот путь среды и автоматически найдет данные сервера, при этом локально вы можете использовать другие пути, просто поменяв `.env` переменную.
