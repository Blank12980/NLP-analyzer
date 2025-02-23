# Emotion Analysis CNN

Этот проект представляет собой анализатор эмоций, использующий сверточные нейронные сети (CNN) для определения эмоциональной окраски текстовых сообщений. 

## Описание

Модель обучена на наборе данных, содержащем текстовые сообщения и соответствующие им эмоции. Цель проекта - классифицировать сообщения на основе их эмоциональной нагрузки, такой как радость, грусть, гнев и т.д.
набор данных взят  с сайта kaggle:
```
https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data
```
## Установка

1. Клонируйте этот репозиторий:
   ```
   gh repo clone Blank12980/NLP-analyzer
   ```

2. Перейдите в директорию проекта:
   ```
   cd emotion-analysis-cnn
   ```

3. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

## Использование

1. добавте файд cfg.py, в который впиишите token = "ваш токен телеграмм бота".
2. Запустите анализатор:
   ```
   python main.py
   ```
   
3. отправте сообщение вашему телеграм боту.

## Примеры

Пример входных данных:
```
"I'm so hate you"
```

Пример выходных данных:
```
emotions = anger 

80 % - angry
9 % - fear
0 % - joy
2 % - love
1 % - sadness
3 % - surprise
```

## Обучение модели

Если вы хотите обучить модель на своих данных, следуйте этим шагам:

1. Подготовьте ваш набор данных в формате CSV с двумя столбцами: `text` и `emotion`.
2. скачайте зависимости из файла ./src/requirements.txt
   ```
   cd ./src/
   pip install -r requirements.txt
   ```
3. Запустите скрипт обучения:
   ```
   python ./scr/functions.py --data ./pathToFile/file.csv
   ```

## Контакты

Для вопросов вы не сможете связываться по электронной почте
