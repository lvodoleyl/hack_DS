# NLP-сервис для Хакатона.
Данный сервис производит постобработку текстов, полченных
после ASR-модели. 

Сервис написан на основе микрофреймворка __Flask__.

Последовательно обработки:
1. Исправление ошибок в словах после ASR, с помощью __Yandex Speller__.
2. Расставление знаков (пунктуация), с помощью предобученной модели - __snakers4/silero-models__.
3. Обучение Тематической модели __LDA__ для извлечения тематик текста и главных слов каждой темы- тэгов. 
4. Применение предобученного NER в библиотеке Natasha для извлечения имен собственных, организаций, локаций и 
отображение их в тексте в виде тэгов \<LOC>, \<ORG>, \<PER>.
5. Применение статистического алгоритма - модели TextRank из библиотеки Gensim - 
для __"извлекающей" аннотации__ текста.

На вход по route "/" json {"text": string}.

На выходе json:
1. {"text": string, "annotation": string, "tags": [string]} - при успешной обработке.
2. {"error": string} - при возникновении ошибки.


Для запуска можно использовать
```python main.py```.