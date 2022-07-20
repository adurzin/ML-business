# Интеграция. Итоговый проект

## Тема: определение токсичности комментария

### Работа программы.
    
    Программа определяет насколько токсичен введенный текст комментария.
    Программа обучена и работает на английском языке.
    В папке models находится файл .dill пайплайна обработки комментария с помощью TfidfVectorizer и создание модели на основе логистической регрессии, так как существенных изменений при кросс-валидации более сложных моделей замечено не было.
    Также в папке models находится файл .ipynb с исходным кодом модели.
    Работа:
        - в корневой папке запускаем сервер - run_server_app.py.
        - запускаем файл request.py
        - вводим/копируем комментарий в рабочее приложение
        - получаем результат в % насколько токсичен ваш комментарий.
        - по окончанию работы закрыть приложение с сервером.