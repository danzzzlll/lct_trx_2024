Решение кейса 12 Сбер "Предиктивная модель для рекомендации продукта банка" команды Мисисково

Общая структура решения:
 - AggFeatureSeqEncoder - содержит создание агрегированных признаков для транзакций и гео
 - custom_CoLES - содержится реализация для masked CoLes, CoLES + time features, CoLES + AR, также модуль pandas_preprocessor.py, который включает в себя улучшенную и оптимизированную версию PandasDataPreprocessor из pytorch-lifestream
 - dialog_embs - модуль для агрегации эмбеддингов диалогов по клиентам и месяцам
 - downstream_models - различные downstream модели для объединения модальностей и предсказания по тесту
 - aggregation_model - построение аггрегационных признаков с помощью pivot_tables

 Подробнее об экспериментах:
    1. custom CoLES: masked CoLES - маскирует рандомные транзакции и пытается их восстановить
    2. CoLES + AR - то же самое, что CoLES, но пытается предсказывать следующую транзакцию
    3. AggFeatureSeqEncoder + CoLES - считаем аггрегированные признаки для объединения с эмбеддингами полученными от CoLES.
    4. Получение отдельных агрегированных признаков с помощью pivot_tables
    5. Различная обработка диалоговых эмбеддингов - агрегирование по пользователям, месяцам (среднее, сумма, стандартное отклонение и медиана)
    6. Обучение на истории для пользователей из теста
    7. Обучение на истории для пользователей из теста + из трейна
    7. Запуск LightAutoML совместно с бустингами на предсказание таргетов

Валидация:
    0. Разработали уникальный алгоритм и назвали его Holdout валидация.
    1. История теста для валидации.
    2. При добавлении истории теста в треин - берем последний месяц для каждого клиента.

mean_target: 0.82
target_1: 0.72
target_2: 0.85
target_3: 0.77
target_4: 0.87

Лучшим показало себя решение с объединением всех трех модальностей, средние эмбеддинги диалогов.

