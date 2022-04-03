Документация может быть найдена [здесь](https://text-categorization.readthedocs.io/ru/latest/).

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
python -m ap.train.server --models={path_to_models} --data={path_to_data}
```
где `path_to_models` - путь к каталогу для хранения обученных моделей, `path_to_data` - путь к каталогу с данными.

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

Данные для обучения хранятся внутри директории, переданной в аргументе `data`,` в следующих папках:
* `vw` - полная коллекция документов для обучения в формате VopalWabbit
* `batches` - полная коллекция документов для обучения в формате батчей bigARTM
* `vw_new` - новые документы, добавленные методом `AddDocumentsToCollection`, которые еще не использовались для обучения моделей
* `batches_new` - новые батчи, созданные из документов в папке `vw_new` во время вызова `TrainModel`
#Подготовка образа виртуальной машины для запуска сервисов
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
