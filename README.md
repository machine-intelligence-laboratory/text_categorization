# Документация
Документация может быть найдена [здесь](https://text-categorization.readthedocs.io/ru/documentation/).

# Запуск приложения
## Локальный запуск
Перед локальным запуском необходимо:
1. Установить зависимости, выполнив команду
```bash
pip install -r requirements.txt
```
2. Сгенерировать серверный код GRPC из proto файлов, выполнив скрипт `generate.sh`:
```bash
./generate.sh
```
Проверить правильность окружения можно, выполнив pytest тесты. Для этого необходимо поставить тестовые зависимости:
```bash
pip install -r test_requirements.txt
```
И запустиь тесты:
```bash
python -m pytest ./tests
```
Для локального запуска сервиса получения векторных представлений, выполните команду
```bash
python -m ap.inference.server --model={path_to_model}
```
где `path_to_model` - путь к каталогу с обученной моделью bigARTM.

Для локального запуска сервиса обучения моделей запустите
```bash
python -m ap.train.server --config={path_to_config} --data={path_to_data}
```
где `path_to_config` - путь к конфигу для обучения модели в формате yaml, `path_to_data` - путь к каталогу с данными.

## Запуск в docker-контейнере
Для запуска сервиса получения векторных представлений в docker-контейнере, выполните команду
```bash
docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up
```
Путь к каталогу с обученной моделью должен быть задан в переменной окружения с названием `MODEL_PATH`.
​
Для запуска сервиса обучения моделей запустите
```bash
docker-compose -f docker-compose-train.yml build && docker-compose -f docker-compose-train.yml up
```
Путь к каталогу для хранения обученных моделей должен быть задан в переменной окружения с названием `MODELS_PATH`, а путь к каталогу с данными должен быть задан в переменной окружения с названием `DATA_PATH`.
# Структура папок для сервера обучения
Для хранения данных и обученных моделей в сервере обучения предусмотрена особая структура файлов.
Обученные модели хранятся в каталоге, переданном в аргументе `models`. Каждая модель хранится в папке с названием, соответствующим времени запуска обучения, в формате `YYYYMMDD_HHmmSS`.

Данные для обучения хранятся внутри директории, переданной в аргументе `data`. Директория должна иметь следующую структуру:
```
data
├── BPE_models
│   ├── bpe_model_af_120k.model
│   ├── ...
│   └── bpe_model_zh_120k.model
├── dictionary_train_BPE_wiki_100_120k_11k.txt
├── rubrics_train_grnti.json
├── train_with_combined_grnti_69_rubric.txt
├── udk_codes.json
└── wiki_100
    ├── batches_train
    └── modality_distribution.yml
```

* `BPE_models` - папка с обученными BPE моделями.
    * Модели должны называться в формате `bpe_model_{lang}_120k.model`
* `dictionary_train_BPE_wiki_100_120k_11k.txt` - словарь тематической модели
* `rubrics_train_grnti.json` - файл с ГРНТИ рубриками документов
* `train_with_combined_grnti_69_rubric.txt` - файл с данными для обучения в Vowpal Wabbit формате
* `udk_codes.json` - файл с УДК рубриками документов
* `wiki_100` - папка с данными Wikipedia
    * `batches_train` - папка батчами, предварительно построенными по данным Wikipedia
    * `modality_distribution.yml` - данные о распределении документов Wikipedia по языкам


# Подготовка образа виртуальной машины для запуска сервисов
Для запуска сервиса в составе стандартной виртуальной машины подготовлены скрипты `inference.sh` и `train.sh`. 

Скрипты рассчитаны на образ виртуальной машины с ОС Ubuntu и установленными Docker и Docker Compose.

Так же необходимо поставить Git и клонировать репозиторий с ПО:
```bash 
git clone git@github.com:machine-intelligence-laboratory/text_categorization.git
```
 
Имя пользователя, под которым запускаются сервисы, должно быть `antiplagiat`.

Внутри домашней директории пользователя `/home/antiplagiat` для запуска сервиса получения векторных представлений должна быть создана папка `/home/antiplagiat/models/{MODEL_NAME}`. Название модели может меняться в зависимости от поставленной модели. Соответственным образом необходимо менять значение переменной `MODEL_PATH` в скрипте запуска после установки сервиса.

Внутри домашней директории пользователя `/home/antiplagiat` для запуска сервиса обучения должны быть папки `data` и `models`. В папку `data` необходимо распаковать архив с переданными изначальными данными для обучения модели:
```bash
tar -C /home/antiplagiat/data -xvzf data.tar.gz 
```
Так же в папку `/home/antiplagiat/models/{CURRENT_DATE}` можно положить изначальную модель, которая будет дообучаться новыми данными.
