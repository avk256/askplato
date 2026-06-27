#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_plato_agentic_rag.py

Streamlit-застосунок для Plato Agentic RAG + OpenRouter.
Є графічним аналогом run_plato_agentic_rag.py.

Передумови:
1. Поруч із цим файлом має бути plato_agentic_rag_engine.py.
2. Має бути готова папка індексу етапу 2:

   plato_vector_index_bge_m3/
   ├── plato_bge_m3.faiss
   ├── plato_chunks_metadata.jsonl
   ├── plato_bm25.pkl
   └── index_info.json

3. OpenRouter API key можна задати одним із трьох способів:
   - змінна середовища OPENROUTER_API_KEY;
   - .streamlit/secrets.toml:
         OPENROUTER_API_KEY = "..."
   - поле введення API key у sidebar застосунку.

Запуск:
    streamlit run streamlit_plato_agentic_rag.py

Для CPU-only запуску:
    CUDA_VISIBLE_DEVICES="" streamlit run streamlit_plato_agentic_rag.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# Важливо: це має бути до імпорту torch/sentence-transformers через engine.
# Фінальна LLM працює через OpenRouter, тому GPU на хості не потрібен.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pandas as pd
import streamlit as st

from plato_agentic_rag_engine import (
    ANSWER_MODES,
    MODEL_ALIASES,
    AgenticRAGConfig,
    PlatoAgenticRAGEngine,
    normalize_model_name,
)


REQUIRED_INDEX_FILES = [
    "plato_bge_m3.faiss",
    "plato_chunks_metadata.jsonl",
    "plato_bm25.pkl",
]

DEFAULT_QUESTIONS = [
    "Чому Сократ не боїться смерті?",
    "Порівняй, як Сократ і Протагор розуміють чесноту.",
    "Як у Федоні пояснюється безсмертя душі?",
    "Чому Сократ не хоче тікати з в'язниці у Крітоні?",
    "Порівняй розуміння душі у Федоні та Федрі.",
]

MODE_LABELS = {
    "short": "short — коротка відповідь",
    "detailed": "detailed — розгорнуте пояснення",
    "academic": "academic — академічний стиль",
    "student": "student — простіше пояснення",
    "compare": "compare — порівняння",
}


# -----------------------------
# Utility functions
# -----------------------------

def safe_secrets_get(name: str) -> str | None:
    """Повертає значення зі st.secrets, якщо воно існує."""
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def api_key_fingerprint(api_key: str | None) -> str:
    """Хеш для cache key без розкриття самого API key."""
    if not api_key:
        return "no_key"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def check_index_dir(index_dir: str) -> tuple[bool, list[str]]:
    """Перевіряє наявність потрібних файлів індексу."""
    path = Path(index_dir)
    if not path.exists():
        return False, [f"Папку індексу не знайдено: {path}"]

    missing = [str(path / name) for name in REQUIRED_INDEX_FILES if not (path / name).exists()]
    if missing:
        return False, ["Відсутні потрібні файли індексу:"] + missing

    return True, []


def source_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Перетворює sources у компактні рядки для таблиці."""
    rows: list[dict[str, Any]] = []
    for i, src in enumerate(sources or [], start=1):
        rows.append(
            {
                "#": i,
                "dialogue": src.get("dialogue", ""),
                "pages": f"{src.get('page_start', '?')}-{src.get('page_end', '?')}",
                "chunk_id": src.get("chunk_id", ""),
                "rerank_score": round(float(src.get("rerank_score", 0.0)), 4),
                "subquery": src.get("subquery", ""),
            }
        )
    return rows



def get_display_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Повертає джерела для відображення в UI.

    Важливо: result["sources"] у plato_agentic_rag_engine.py навмисно є компактним
    списком без повного поля "text", щоб не роздувати лог. Повний текст chunk-ів
    зберігається в result["retrieval"]["results"]. Тому тут зшиваємо компактні
    джерела з повними retrieval results за chunk_id.
    """
    compact_sources = result.get("sources") or []
    retrieval = result.get("retrieval") or {}
    full_sources = retrieval.get("results") or []

    if not compact_sources:
        return full_sources

    full_by_chunk_id = {
        str(src.get("chunk_id")): src
        for src in full_sources
        if src.get("chunk_id") is not None
    }

    merged: list[dict[str, Any]] = []
    for src in compact_sources:
        chunk_id = str(src.get("chunk_id"))
        full = full_by_chunk_id.get(chunk_id, {})
        item = full.copy()
        item.update(src)
        # Гарантуємо, що повний текст не буде втрачений через compact source.
        if not item.get("text") and full.get("text"):
            item["text"] = full.get("text")
        merged.append(item)

    return merged


def retrieval_round_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = result.get("retrieval") or {}
    trace = retrieval.get("agent_trace") or {}
    rounds = trace.get("rounds") or []
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(rounds, start=1):
        rows.append(
            {
                "#": i,
                "threshold": item.get("threshold"),
                "n_hits": item.get("n_hits"),
                "retry": bool(item.get("retry", False)),
                "subquery": item.get("subquery", ""),
            }
        )
    return rows


def render_agent_trace(result: dict[str, Any]) -> None:
    """Виводить agent trace у Streamlit."""
    retrieval = result.get("retrieval") or {}
    plan = retrieval.get("plan") or {}
    sufficiency = retrieval.get("sufficiency") or {}

    st.subheader("Agent trace")

    col1, col2, col3 = st.columns(3)
    col1.metric("Intent", str(plan.get("intent", "—")))
    col2.metric("Needs comparison", str(plan.get("needs_comparison", "—")))
    col3.metric("Dialogues", str(sufficiency.get("n_dialogues", "—")))

    st.markdown("**Mentioned dialogues:**")
    mentioned = plan.get("mentioned_dialogues") or []
    st.write(", ".join(mentioned) if mentioned else "—")

    st.markdown("**Subqueries:**")
    subqueries = plan.get("subqueries") or []
    if subqueries:
        for i, subq in enumerate(subqueries, start=1):
            st.write(f"{i}. {subq}")
    else:
        st.write("—")

    rows = retrieval_round_rows(result)
    st.markdown("**Retrieval rounds:**")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("—")

    st.markdown("**Sufficiency:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("is_sufficient", str(sufficiency.get("is_sufficient", "—")))
    col2.metric("n_sources", str(sufficiency.get("n_sources", "—")))
    col3.metric("max_score", f"{float(sufficiency.get('max_score', 0.0)):.4f}")
    col4.metric("mean_score", f"{float(sufficiency.get('mean_score', 0.0)):.4f}")

    reasons = sufficiency.get("reasons") or []
    if reasons:
        st.warning("\n".join(f"- {r}" for r in reasons))


@st.cache_resource(show_spinner=False)
def load_engine(
    index_dir: str,
    embedding_model_name: str,
    reranker_model_name: str,
    output_dir: str,
    api_key_hash: str,
) -> PlatoAgenticRAGEngine:
    """
    Завантажує engine один раз і кешує його між rerun Streamlit.

    api_key_hash використовується тільки як частина cache key. Сам API key не зберігається тут.
    """
    _ = api_key_hash
    cfg = AgenticRAGConfig(
        index_dir=index_dir,
        embedding_model_name=embedding_model_name,
        reranker_model_name=reranker_model_name,
        output_dir=output_dir,
    )
    return PlatoAgenticRAGEngine(cfg)


def apply_runtime_config(
    engine: PlatoAgenticRAGEngine,
    *,
    model_slug: str,
    vector_top_k: int,
    bm25_top_k: int,
    candidate_top_k: int,
    rerank_top_k: int,
    final_top_k: int,
    retry_final_top_k: int,
    relevance_threshold: float,
    retry_relevance_threshold: float,
    min_sources: int,
    min_compare_dialogues: int,
    max_context_chars: int,
    max_subqueries: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: int,
    save_agent_trace: bool,
) -> None:
    """Оновлює параметри cfg без повторного завантаження моделей."""
    cfg = engine.cfg
    cfg.openrouter_model = model_slug
    cfg.vector_top_k = vector_top_k
    cfg.bm25_top_k = bm25_top_k
    cfg.candidate_top_k = candidate_top_k
    cfg.rerank_top_k = rerank_top_k
    cfg.final_top_k = final_top_k
    cfg.retry_final_top_k = retry_final_top_k
    cfg.relevance_threshold = relevance_threshold
    cfg.retry_relevance_threshold = retry_relevance_threshold
    cfg.min_sources = min_sources
    cfg.min_compare_dialogues = min_compare_dialogues
    cfg.max_context_chars = max_context_chars
    cfg.max_subqueries = max_subqueries
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.max_tokens = max_tokens
    cfg.request_timeout = request_timeout
    cfg.save_agent_trace = save_agent_trace


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Plato Agentic RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Plato Agentic RAG + OpenRouter")
st.caption(
    "Streamlit-інтерфейс до системи відповідей за корпусом Платона: "
    "FAISS + BM25 + reranker + Agentic RAG + Gemma через OpenRouter."
)

with st.sidebar:
    st.header("Налаштування запуску")

    st.subheader("OpenRouter")
    env_key = os.getenv("OPENROUTER_API_KEY")
    secrets_key = safe_secrets_get("OPENROUTER_API_KEY")
    user_key = st.text_input(
        "OpenRouter API key",
        type="password",
        value="",
        help=(
            "Можна залишити порожнім, якщо OPENROUTER_API_KEY вже заданий "
            "у змінних середовища або .streamlit/secrets.toml."
        ),
    ).strip()

    active_key = user_key or env_key or secrets_key
    if user_key:
        os.environ["OPENROUTER_API_KEY"] = user_key
    elif secrets_key and not env_key:
        os.environ["OPENROUTER_API_KEY"] = secrets_key

    if active_key:
        st.success("OpenRouter API key знайдено")
    else:
        st.error("OpenRouter API key не знайдено")

    st.subheader("Модель")
    model_options = list(MODEL_ALIASES.keys()) + ["custom"]
    model_choice = st.selectbox("Модель / alias", model_options, index=model_options.index("gemma4"))
    if model_choice == "custom":
        custom_model = st.text_input(
            "Повний OpenRouter model slug",
            value="google/gemma-3-27b-it",
            help="Наприклад: google/gemma-3-27b-it або google/gemma-4-26b-a4b-it",
        ).strip()
        model_slug = custom_model
    else:
        model_slug = normalize_model_name(model_choice)
    st.caption(f"Буде використано: `{model_slug}`")

    st.subheader("Корпус та локальні моделі")
    index_dir = st.text_input("Папка індексу", value="plato_vector_index_bge_m3")
    embedding_model_name = st.text_input("Embedding model", value="BAAI/bge-m3")
    reranker_model_name = st.text_input("Reranker model", value="BAAI/bge-reranker-v2-m3")
    output_dir = st.text_input("Папка логів", value="plato_agentic_rag_output")

    index_ok, index_errors = check_index_dir(index_dir)
    if index_ok:
        st.success("Індекс знайдено")
    else:
        st.error("Проблема з індексом")
        for err in index_errors:
            st.caption(err)

    st.subheader("Режим відповіді")
    mode_values = list(ANSWER_MODES.keys())
    mode = st.selectbox(
        "mode",
        mode_values,
        index=mode_values.index("academic") if "academic" in mode_values else 0,
        format_func=lambda x: MODE_LABELS.get(x, x),
    )

    with st.expander("Retrieval parameters", expanded=False):
        vector_top_k = st.slider("vector_top_k", 5, 100, 30, 5)
        bm25_top_k = st.slider("bm25_top_k", 5, 100, 30, 5)
        candidate_top_k = st.slider("candidate_top_k", 5, 100, 30, 5)
        rerank_top_k = st.slider("rerank_top_k", 3, 50, 10, 1)
        final_top_k = st.slider("final_top_k", 1, 15, 5, 1)
        retry_final_top_k = st.slider("retry_final_top_k", 1, 20, 8, 1)
        relevance_threshold = st.slider("relevance_threshold", 0.0, 1.0, 0.35, 0.01)
        retry_relevance_threshold = st.slider("retry_relevance_threshold", 0.0, 1.0, 0.25, 0.01)
        min_sources = st.slider("min_sources", 1, 8, 2, 1)
        min_compare_dialogues = st.slider("min_compare_dialogues", 1, 4, 2, 1)
        max_subqueries = st.slider("max_subqueries", 1, 10, 5, 1)
        max_context_chars = st.slider("max_context_chars", 2000, 20000, 9000, 500)

    with st.expander("Generation parameters", expanded=False):
        temperature = st.slider("temperature", 0.0, 1.5, 0.0, 0.05)
        top_p = st.slider("top_p", 0.1, 1.0, 1.0, 0.05)
        max_tokens = st.slider("max_tokens", 100, 3000, 700, 50)
        request_timeout = st.slider("request_timeout, seconds", 10, 300, 90, 10)

    with st.expander("Вивід і логування", expanded=False):
        show_sources = st.checkbox("Показати джерела", value=True)
        show_trace = st.checkbox("Показати Agent trace", value=True)
        log_answer = st.checkbox("Записувати відповіді у JSONL log", value=True)
        save_agent_trace = st.checkbox("Зберігати agent trace у результаті", value=True)

    if st.button("Очистити кеш engine"):
        st.cache_resource.clear()
        st.success("Кеш очищено. Engine буде завантажено заново.")
        st.rerun()


# -----------------------------
# Main interaction area
# -----------------------------

left, right = st.columns([2, 1])

with right:
    st.subheader("Приклади питань")
    selected_example = st.selectbox("Вибрати приклад", ["—"] + DEFAULT_QUESTIONS)

with left:
    initial_question = "" if selected_example == "—" else selected_example
    question = st.text_area(
        "Питання до корпусу Платона",
        value=initial_question,
        height=130,
        placeholder="Наприклад: Порівняй, як Сократ і Протагор розуміють чесноту.",
    )

run_col, status_col = st.columns([1, 3])
with run_col:
    run_button = st.button("🔎 Згенерувати відповідь", type="primary", use_container_width=True)

with status_col:
    st.info(
        "Перший запуск може бути довшим, бо завантажуються локальні embedding/reranking моделі. "
        "Наступні запити працюватимуть швидше завдяки кешу Streamlit."
    )

if not active_key:
    st.warning("Задайте OpenRouter API key у sidebar або через змінну середовища OPENROUTER_API_KEY.")

if not index_ok:
    st.warning("Перевірте шлях до папки `plato_vector_index_bge_m3/` і наявність файлів індексу.")

if run_button:
    question = question.strip()
    if not question:
        st.error("Введіть питання.")
        st.stop()
    if not active_key:
        st.error("OpenRouter API key не задано.")
        st.stop()
    if not index_ok:
        st.error("Індекс не знайдено або він неповний.")
        st.stop()

    api_hash = api_key_fingerprint(active_key)

    try:
        with st.spinner("Завантаження Plato Agentic RAG engine..."):
            engine = load_engine(
                index_dir=index_dir,
                embedding_model_name=embedding_model_name,
                reranker_model_name=reranker_model_name,
                output_dir=output_dir,
                api_key_hash=api_hash,
            )

        apply_runtime_config(
            engine,
            model_slug=model_slug,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            candidate_top_k=candidate_top_k,
            rerank_top_k=rerank_top_k,
            final_top_k=final_top_k,
            retry_final_top_k=retry_final_top_k,
            relevance_threshold=relevance_threshold,
            retry_relevance_threshold=retry_relevance_threshold,
            min_sources=min_sources,
            min_compare_dialogues=min_compare_dialogues,
            max_context_chars=max_context_chars,
            max_subqueries=max_subqueries,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            save_agent_trace=save_agent_trace,
        )

        t0 = time.perf_counter()
        with st.spinner("Agentic retrieval + OpenRouter generation..."):
            result = engine.answer(
                question=question,
                mode=mode,
                model=model_slug,
                log=log_answer,
            )
        wall_time = time.perf_counter() - t0

    except Exception as exc:
        st.error("Помилка під час виконання запиту")
        st.exception(exc)
        st.stop()

    st.session_state["last_result"] = result
    st.session_state["last_wall_time"] = wall_time


# -----------------------------
# Results rendering
# -----------------------------

result = st.session_state.get("last_result")
wall_time = st.session_state.get("last_wall_time")

if result:
    st.divider()
    st.header("Відповідь")

    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("Status", str(result.get("status", "—")))
    meta2.metric("Model", str(result.get("model", "—")).split("/")[-1])
    meta3.metric("elapsed_sec", str(result.get("elapsed_sec", "—")))
    meta4.metric("wall_time", f"{wall_time:.2f} s" if isinstance(wall_time, (int, float)) else "—")

    st.markdown(result.get("answer", ""))

    sources = get_display_sources(result)
    if show_sources:
        with st.expander("Джерела", expanded=True):
            if sources:
                rows = source_rows(sources)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                selected_source = st.selectbox(
                    "Переглянути текст джерела",
                    options=list(range(len(sources))),
                    format_func=lambda i: f"{i + 1}. {sources[i].get('dialogue', '?')} | {sources[i].get('chunk_id', '?')}",
                )
                src = sources[selected_source]
                chunk_id = str(src.get("chunk_id", "unknown"))
                chunk_text = src.get("text") or src.get("clean_text") or src.get("content") or ""
                st.markdown(
                    f"**{src.get('dialogue', '?')}**, сторінки PDF "
                    f"{src.get('page_start', '?')}-{src.get('page_end', '?')}, "
                    f"chunk_id=`{chunk_id}`"
                )
                if chunk_text:
                    st.text_area(
                        "Текст chunk-а",
                        value=chunk_text,
                        height=300,
                        key=f"chunk_text_{chunk_id}_{selected_source}",
                        disabled=True,
                    )
                else:
                    st.warning(
                        "Текст chunk-а відсутній у compact `sources`. "
                        "Перевірте, що в результаті є `retrieval.results` і що `save_agent_trace=True`."
                    )
                    st.json(src)
            else:
                st.info("Джерела не повернуті або відповідь має статус low_confidence.")

    if show_trace:
        with st.expander("Agent trace", expanded=True):
            render_agent_trace(result)

    with st.expander("JSON результату", expanded=False):
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        st.code(result_json, language="json")
        st.download_button(
            label="Завантажити JSON результату",
            data=result_json.encode("utf-8"),
            file_name="plato_agentic_rag_result.json",
            mime="application/json",
        )
else:
    st.divider()
    st.info("Введіть питання і натисніть **Згенерувати відповідь**.")
