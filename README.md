# AskPlato

Проєкт реалізує інтелектуальну систему відповідей на запитання про твори Платона на основі підходу Agentic RAG. Система попередньо обробляє корпус текстів, розбиває його на фрагменти, створює векторний та лексичний індекси для пошуку, а під час запиту користувача аналізує тип питання, формує підзапити, знаходить релевантні джерела, переоцінює їх за допомогою reranker-моделі та перевіряє достатність знайденого контексту. Після цього відібрані фрагменти передаються до мовної моделі через OpenRouter для генерації остаточної відповіді українською мовою. На відміну від звичайного RAG, проєкт використовує агентну логіку пошуку: планування, комбінування FAISS і BM25, повторний retrieval за потреби, аналіз якості джерел і прозорий вивід використаних фрагментів. У результаті система дозволяє ставити питання до корпусу Платона, отримувати обґрунтовані відповіді з посиланням на джерела та змінювати модель генерації, наприклад Gemma, GPT, Gemini або Mistral, без локального донавчання LLM.


Узагальнений алгоритм проєкту

1. Взяти PDF із творами Платона.

2. Витягти текст із PDF.

3. Очистити текст:
   - прибрати службові символи;
   - нормалізувати пробіли;
   - нормалізувати імена мовців;
   - прибрати номери сторінок і зайві маркери.

4. Розділити корпус за діалогами.

5. Розбити тексти на chunk-и з overlap.

6. Для кожного chunk-а зберегти:
   - текст;
   - назву діалогу;
   - сторінки PDF;
   - мовців;
   - chunk_id;
   - додаткові метадані.

7. Обчислити embeddings для chunk-ів через BAAI/bge-m3.

8. Побудувати FAISS-індекс.

9. Побудувати BM25-індекс.

10. Зберегти готову папку plato_vector_index_bge_m3/.

11. При запуску системи завантажити:
    - FAISS;
    - BM25;
    - metadata;
    - embedding-модель;
    - reranker;
    - OpenRouter client.

12. Отримати питання користувача.

13. Визначити intent питання.

14. Сформувати підзапити.

15. Для кожного підзапиту виконати:
    - FAISS search;
    - BM25 search;
    - RRF fusion;
    - reranking;
    - threshold filtering.

16. Об’єднати результати всіх підзапитів.

17. Прибрати дублікати і схожі chunk-и.

18. Перевірити достатність джерел.

19. Якщо джерел недостатньо:
    - знизити threshold;
    - збільшити кількість кандидатів;
    - повторити retrieval.

20. Сформувати фінальний RAG-контекст.

21. Сформувати prompt для OpenRouter-моделі.

22. Надіслати запит до Gemma / GPT / Gemini / Mistral.

23. Отримати відповідь.

24. Перевірити, чи є у відповіді джерела.

25. Якщо джерел немає — автоматично додати їх.

26. Повернути користувачу:
    - відповідь;
    - список джерел;
    - agent trace;
    - службову інформацію про retrieval.

Для відповіді потрібні тільки артефакти етапу векторизації chunk-ів:

```text
plato_vector_index_bge_m3/
├── plato_bge_m3.faiss
├── plato_chunks_metadata.jsonl
├── plato_bm25.pkl
└── index_info.json
```

## Встановлення

```bash
pip install -r requirements_agentic_rag.txt
```

## OpenRouter API key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

У Colab:

```python
import os
from getpass import getpass
os.environ["OPENROUTER_API_KEY"] = getpass("OpenRouter API key: ")
```

## Приклад запуску

```python
from plato_agentic_rag_engine import PlatoAgenticRAGEngine, AgenticRAGConfig

cfg = AgenticRAGConfig(
    index_dir="plato_vector_index_bge_m3",
    openrouter_model="google/gemma-4-31b-it:free",  # або google/gemma-3-27b-it
)

engine = PlatoAgenticRAGEngine(cfg)
result = engine.answer("Чому Сократ не боїться смерті?", mode="academic")
print(result["answer"])
```

## Моделі OpenRouter

Можна передати alias або повний model slug:

```python
result = engine.answer("Що Сократ говорить про душу?", model="gemma3")
result = engine.answer("Що Сократ говорить про душу?", model="gemma4")
result = engine.answer("Що Сократ говорить про душу?", model="google/gemma-3-12b-it")
```

Вбудовані alias:

- `gemma3` → `google/gemma-3-27b-it`
- `gemma3_free` → `google/gemma-3-27b-it:free`
- `gemma3_12b` → `google/gemma-3-12b-it`
- `gemma3_4b` → `google/gemma-3-4b-it`
- `gemma4` → `google/gemma-4-31b-it:free`
- `gemma4_26b` → `google/gemma-4-26b-a4b-it`

## Що робить Agentic RAG

1. Аналізує запит: тип питання, згадані діалоги, потреба в порівнянні.
2. Формує кілька підзапитів.
3. Запускає retrieval tools: FAISS, BM25, RRF, reranker.
4. Перевіряє достатність джерел.
5. Якщо джерел мало, повторює retrieval з ширшими параметрами.
6. Прибирає дублікати й близькі фрагменти.
7. Формує контекст і source audit.
8. Передає остаточний prompt у Gemma через OpenRouter.
9. Примусово додає джерела, якщо модель їх пропустила.
