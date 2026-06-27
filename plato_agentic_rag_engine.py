#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plato_agentic_rag_engine.py

Agentic RAG engine for Ukrainian QA over Plato dialogues.

Main changes compared with the previous Stage 4 notebook:
- No Stage 3 / LoRA / QLoRA adapter is required at inference time.
- Retrieval still uses Stage 2 artifacts: FAISS + metadata + BM25.
- The final answer is generated through OpenRouter, for example with Gemma 3 or Gemma 4.
- The retrieval controller is agentic: it plans subqueries, executes retrieval tools,
  checks source sufficiency, optionally retries with wider retrieval, and then generates
  the final answer only from verified context.

Expected input structure:

plato_vector_index_bge_m3/
├── plato_bge_m3.faiss
├── plato_chunks_metadata.jsonl
├── plato_bm25.pkl
└── index_info.json              # optional

Environment variable:
    OPENROUTER_API_KEY="..."

Minimal usage:

    from plato_agentic_rag_engine import PlatoAgenticRAGEngine, AgenticRAGConfig

    cfg = AgenticRAGConfig(
        index_dir="plato_vector_index_bge_m3",
        openrouter_model="google/gemma-4-31b-it:free",
    )
    engine = PlatoAgenticRAGEngine(cfg)
    result = engine.answer("Чому Сократ не боїться смерті?", mode="academic")
    print(result["answer"])
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import requests
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9']+")

PLATO_DIALOGUES = [
    "Апологія Сократа",
    "Крітон",
    "Іон",
    "Протагор",
    "Федон",
    "Федр",
]

ANSWER_MODES = {
    "short": "Дай стислу відповідь українською у 3–6 реченнях. Не додавай інформації поза CONTEXT.",
    "detailed": "Дай розгорнуте пояснення українською. Поясни головні аргументи, але використовуй тільки CONTEXT.",
    "academic": "Відповідай у діловому академічному стилі. Структуруй відповідь: теза, пояснення на основі тексту, можлива інтерпретація, джерела.",
    "student": "Поясни простішою українською мовою, ніби для студента, але не втрачай зв'язок із CONTEXT.",
    "compare": "Якщо CONTEXT містить фрагменти з різних діалогів або різних позицій, порівняй їх. Якщо підстав для порівняння недостатньо, прямо скажи про це.",
}

MODEL_ALIASES = {
    # Google Gemma
    "gemma3": "google/gemma-3-27b-it",
    "gemma3_free": "google/gemma-3-27b-it:free",
    "gemma3_27b": "google/gemma-3-27b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma4": "google/gemma-4-31b-it:free",
    "gemma4_free": "google/gemma-4-31b-it:free",
    "gemma4_31b_free": "google/gemma-4-31b-it:free",
    "gemma4_26b": "google/gemma-4-26b-a4b-it",

    # OpenAI / GPT через OpenRouter
    "gpt_4o_mini": "openai/gpt-4o-mini",
    "gpt_4o": "openai/gpt-4o",
    "gpt_latest": "openai/gpt-latest",

    # Google Gemini через OpenRouter
    "gemini_flash": "google/gemini-2.5-flash",
    "gemini_pro": "google/gemini-2.5-pro",

    # Mistral через OpenRouter
    "mistral_small": "mistralai/mistral-small-3.2-24b-instruct",
    "mistral_medium": "mistralai/mistral-medium-3.1",
    "mistral_large": "mistralai/mistral-large",

    # Anthropic
    "claude_sonnet": "anthropic/claude-3.5-sonnet",

    # Meta Llama
    "llama_70b": "meta-llama/llama-3.3-70b-instruct",
}

@dataclass
class AgenticRAGConfig:
    # Stage 2 artifacts
    index_dir: str = "plato_vector_index_bge_m3"
    faiss_index_file: str = "plato_bge_m3.faiss"
    metadata_file: str = "plato_chunks_metadata.jsonl"
    bm25_file: str = "plato_bm25.pkl"
    index_info_file: str = "index_info.json"

    # Local retrieval models
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 1024

    # Agentic retrieval parameters
    vector_top_k: int = 30
    bm25_top_k: int = 30
    candidate_top_k: int = 30
    rerank_top_k: int = 10
    final_top_k: int = 5
    retry_final_top_k: int = 8
    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    relevance_threshold: float = 0.35
    retry_relevance_threshold: float = 0.25
    min_sources: int = 2
    min_compare_dialogues: int = 2
    max_context_chars: int = 9000
    max_subqueries: int = 5
    diversity_min_token_jaccard_distance: float = 0.18

    # OpenRouter generation
    openrouter_model: str = "google/gemma-4-31b-it:free"
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_http_referer: Optional[str] = None
    openrouter_app_title: str = "Plato Agentic RAG"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 700
    request_timeout: int = 90

    # Logs
    output_dir: str = "plato_agentic_rag_output"
    log_jsonl: str = "answer_logs.jsonl"
    save_agent_trace: bool = True


@dataclass
class AgentPlan:
    original_question: str
    intent: str
    mentioned_dialogues: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    needs_comparison: bool = False


@dataclass
class SufficiencyReport:
    is_sufficient: bool
    n_sources: int
    max_score: float
    mean_score: float
    covered_subqueries: int
    n_dialogues: int
    reasons: list[str] = field(default_factory=list)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tokenize_uk(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def normalize_model_name(model_or_alias: str) -> str:
    return MODEL_ALIASES.get(model_or_alias, model_or_alias)


def text_jaccard(a: str, b: str) -> float:
    ta = set(tokenize_uk(a))
    tb = set(tokenize_uk(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class SafeBGEReranker:
    def __init__(self, model_name: str, device: str | None = None, max_length: int = 1024):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        query: str,
        passages: list[str],
        batch_size: int = 4,
        apply_sigmoid: bool = True,
    ) -> list[float]:
        if not passages:
            return []

        all_scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            batch_passages = passages[start : start + batch_size]
            encoded = self.tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            logits = outputs.logits
            raw_scores = logits.view(-1) if logits.shape[-1] == 1 else logits[:, -1]
            scores = torch.sigmoid(raw_scores) if apply_sigmoid else raw_scores
            all_scores.extend(scores.detach().cpu().float().tolist())
        return all_scores


class OpenRouterGenerator:
    def __init__(self, cfg: AgenticRAGConfig):
        self.cfg = cfg
        api_key = os.getenv(cfg.openrouter_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Не знайдено змінну середовища {cfg.openrouter_api_key_env}. "
                "Перед запуском задайте OPENROUTER_API_KEY."
            )
        self.api_key = api_key

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        model_name = normalize_model_name(model or self.cfg.openrouter_model)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.cfg.openrouter_app_title,
        }
        if self.cfg.openrouter_http_referer:
            headers["HTTP-Referer"] = self.cfg.openrouter_http_referer

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }

        response = requests.post(
            self.cfg.openrouter_base_url,
            headers=headers,
            json=payload,
            timeout=self.cfg.request_timeout,
        )
        if not response.ok:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:1000]}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise RuntimeError(f"Неочікуваний формат відповіді OpenRouter: {data}") from exc


class PlatoAgenticRAGEngine:
    def __init__(self, cfg: AgenticRAGConfig):
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)
        self.faiss_index_path = self.index_dir / cfg.faiss_index_file
        self.metadata_path = self.index_dir / cfg.metadata_file
        self.bm25_path = self.index_dir / cfg.bm25_file
        self.index_info_path = self.index_dir / cfg.index_info_file
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / cfg.log_jsonl

        self._check_required_files()
        self.index = faiss.read_index(str(self.faiss_index_path))
        self.metadata = read_jsonl(self.metadata_path)
        with self.bm25_path.open("rb") as f:
            bm25_payload = pickle.load(f)
        self.bm25 = bm25_payload["bm25"] if isinstance(bm25_payload, dict) and "bm25" in bm25_payload else bm25_payload

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS має {self.index.ntotal} векторів, а metadata має {len(self.metadata)} записів."
            )

        self.embedding_model = SentenceTransformer(cfg.embedding_model_name)
        self.reranker = SafeBGEReranker(cfg.reranker_model_name, max_length=cfg.reranker_max_length)
        self.generator = OpenRouterGenerator(cfg)

    def _check_required_files(self) -> None:
        missing = []
        for label, path in {
            "FAISS index": self.faiss_index_path,
            "Metadata JSONL": self.metadata_path,
            "BM25 pickle": self.bm25_path,
        }.items():
            if not path.exists():
                missing.append((label, path))
        if missing:
            msg = "Відсутні потрібні файли етапу 2:\n" + "\n".join(f"- {label}: {path}" for label, path in missing)
            raise FileNotFoundError(msg)

    # -----------------------------
    # Agent: plan
    # -----------------------------
    def make_plan(self, question: str) -> AgentPlan:
        q = question.strip()
        q_lower = q.lower()

        mentioned = [d for d in PLATO_DIALOGUES if d.lower() in q_lower]
        compare_words = ["порівняй", "порівняти", "відмінність", "спільне", "різниця", "між", "зістав"]
        citation_words = ["де", "цитата", "цитує", "сторін", "фрагмент", "джерело"]
        interpretation_words = ["чому", "як", "поясни", "сенс", "значення", "інтерпрета"]

        needs_comparison = any(w in q_lower for w in compare_words)
        if needs_comparison:
            intent = "compare"
        elif any(w in q_lower for w in citation_words):
            intent = "evidence_lookup"
        elif any(w in q_lower for w in interpretation_words):
            intent = "interpretation"
        else:
            intent = "direct_answer"

        subqueries = [q]

        # Просте розбиття для порівняльних або складених запитів.
        split_candidates = re.split(r"\s+(?:і|та|або|але|проте|порівняно з|у порівнянні з)\s+", q, flags=re.IGNORECASE)
        for part in split_candidates:
            part = part.strip(" ,.;:-")
            if len(part) >= 12 and part.lower() != q_lower:
                subqueries.append(part)

        # Якщо згадано конкретні діалоги, робимо окремі підзапити з цими назвами.
        for dialogue in mentioned:
            subqueries.append(f"{q} {dialogue}")

        # Для типових питань про Сократа додаємо варіант з іменем, якщо його немає.
        if "сократ" not in q_lower and any(w in q_lower for w in ["смерт", "душ", "чеснот", "справедлив", "втекти", "в'язниц"]):
            subqueries.append(f"{q} Сократ")

        # Дедуплікація зі збереженням порядку.
        unique: list[str] = []
        seen: set[str] = set()
        for s in subqueries:
            key = s.lower()
            if key not in seen:
                unique.append(s)
                seen.add(key)

        return AgentPlan(
            original_question=q,
            intent=intent,
            mentioned_dialogues=mentioned,
            subqueries=unique[: self.cfg.max_subqueries],
            needs_comparison=needs_comparison,
        )

    # -----------------------------
    # Retrieval tools
    # -----------------------------
    def vector_search(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        k = top_k or self.cfg.vector_top_k
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            item = self.metadata[int(idx)].copy()
            item["vector_id"] = int(idx)
            item["vector_rank"] = rank
            item["vector_score"] = float(score)
            results.append(item)
        return results

    def bm25_search(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        k = top_k or self.cfg.bm25_top_k
        query_tokens = tokenize_uk(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            item = self.metadata[int(idx)].copy()
            item["vector_id"] = int(idx)
            item["bm25_rank"] = rank
            item["bm25_score"] = score
            results.append(item)
        return results

    def reciprocal_rank_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self.cfg.candidate_top_k
        fused_scores: dict[int, float] = {}
        details: dict[int, dict[str, Any]] = {}

        for rank, item in enumerate(vector_results, start=1):
            vid = int(item["vector_id"])
            fused_scores[vid] = fused_scores.get(vid, 0.0) + self.cfg.vector_weight / (self.cfg.rrf_k + rank)
            details.setdefault(vid, self.metadata[vid].copy())
            details[vid]["vector_id"] = vid
            details[vid]["vector_rank"] = rank
            details[vid]["vector_score"] = item.get("vector_score")

        for rank, item in enumerate(bm25_results, start=1):
            vid = int(item["vector_id"])
            fused_scores[vid] = fused_scores.get(vid, 0.0) + self.cfg.bm25_weight / (self.cfg.rrf_k + rank)
            details.setdefault(vid, self.metadata[vid].copy())
            details[vid]["vector_id"] = vid
            details[vid]["bm25_rank"] = rank
            details[vid]["bm25_score"] = item.get("bm25_score")

        fused = []
        for vid, score in fused_scores.items():
            item = details[vid].copy()
            item["hybrid_score"] = float(score)
            fused.append(item)
        return sorted(fused, key=lambda x: x["hybrid_score"], reverse=True)[:k]

    def hybrid_search(self, query: str, vector_top_k: Optional[int] = None, bm25_top_k: Optional[int] = None, candidate_top_k: Optional[int] = None) -> list[dict[str, Any]]:
        vector_results = self.vector_search(query, top_k=vector_top_k or self.cfg.vector_top_k)
        bm25_results = self.bm25_search(query, top_k=bm25_top_k or self.cfg.bm25_top_k)
        return self.reciprocal_rank_fusion(vector_results, bm25_results, top_k=candidate_top_k or self.cfg.candidate_top_k)

    def rerank_candidates(self, query: str, candidates: list[dict[str, Any]], top_k: Optional[int] = None) -> list[dict[str, Any]]:
        k = top_k or self.cfg.rerank_top_k
        passages = [item.get("text", "") for item in candidates]
        scores = self.reranker.score(query=query, passages=passages, batch_size=4, apply_sigmoid=True)
        reranked = []
        for item, score in zip(candidates, scores):
            new_item = item.copy()
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)
        return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:k]

    def retrieve_one_subquery(
        self,
        subquery: str,
        plan: AgentPlan,
        threshold: float,
        vector_top_k: int,
        bm25_top_k: int,
        candidate_top_k: int,
        rerank_top_k: int,
    ) -> list[dict[str, Any]]:
        candidates = self.hybrid_search(
            subquery,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            candidate_top_k=candidate_top_k,
        )
        reranked = self.rerank_candidates(subquery, candidates, top_k=rerank_top_k)

        filtered = []
        for item in reranked:
            if item.get("rerank_score", 0.0) < threshold:
                continue
            if plan.mentioned_dialogues and item.get("dialogue") not in plan.mentioned_dialogues:
                # Не відкидаємо повністю, але знижуємо пріоритет: іноді питання згадує діалог,
                # а релевантне пояснення трапляється в іншому фрагменті.
                item = item.copy()
                item["dialogue_mismatch"] = True
                item["rerank_score"] = item.get("rerank_score", 0.0) * 0.85
            item = item.copy()
            item["subquery"] = subquery
            filtered.append(item)
        return filtered

    # -----------------------------
    # Agent: source audit and context
    # -----------------------------
    def evaluate_sufficiency(self, plan: AgentPlan, sources: list[dict[str, Any]]) -> SufficiencyReport:
        scores = [float(s.get("rerank_score", 0.0)) for s in sources]
        max_score = max(scores) if scores else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0
        covered_subqueries = len(set(s.get("subquery") for s in sources if s.get("subquery")))
        n_dialogues = len(set(s.get("dialogue") for s in sources if s.get("dialogue")))

        reasons: list[str] = []
        if len(sources) < self.cfg.min_sources:
            reasons.append("замало джерел після threshold-фільтрації")
        if max_score < self.cfg.relevance_threshold:
            reasons.append("низький максимальний rerank_score")
        if plan.needs_comparison and n_dialogues < self.cfg.min_compare_dialogues:
            reasons.append("для порівняння бажано мати фрагменти принаймні з двох діалогів або позицій")
        if plan.mentioned_dialogues:
            found = set(s.get("dialogue") for s in sources)
            missing = [d for d in plan.mentioned_dialogues if d not in found]
            if missing:
                reasons.append("не знайдено достатніх фрагментів зі згаданих діалогів: " + ", ".join(missing))

        is_sufficient = not reasons
        return SufficiencyReport(
            is_sufficient=is_sufficient,
            n_sources=len(sources),
            max_score=max_score,
            mean_score=mean_score,
            covered_subqueries=covered_subqueries,
            n_dialogues=n_dialogues,
            reasons=reasons,
        )

    def deduplicate_and_diversify(self, sources: list[dict[str, Any]], final_top_k: Optional[int] = None) -> list[dict[str, Any]]:
        k = final_top_k or self.cfg.final_top_k
        sorted_sources = sorted(
            sources,
            key=lambda x: (float(x.get("rerank_score", 0.0)), float(x.get("hybrid_score", 0.0))),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for item in sorted_sources:
            chunk_id = str(item.get("chunk_id"))
            if chunk_id in seen_chunk_ids:
                continue
            text = item.get("text", "")
            too_similar = any(
                text_jaccard(text, s.get("text", "")) > (1 - self.cfg.diversity_min_token_jaccard_distance)
                for s in selected
            )
            if too_similar:
                continue
            selected.append(item)
            seen_chunk_ids.add(chunk_id)
            if len(selected) >= k:
                break
        return selected

    def agentic_retrieve(self, question: str) -> dict[str, Any]:
        plan = self.make_plan(question)
        trace: dict[str, Any] = {
            "plan": asdict(plan),
            "rounds": [],
        }

        all_sources: list[dict[str, Any]] = []
        for subq in plan.subqueries:
            hits = self.retrieve_one_subquery(
                subquery=subq,
                plan=plan,
                threshold=self.cfg.relevance_threshold,
                vector_top_k=self.cfg.vector_top_k,
                bm25_top_k=self.cfg.bm25_top_k,
                candidate_top_k=self.cfg.candidate_top_k,
                rerank_top_k=self.cfg.rerank_top_k,
            )
            all_sources.extend(hits)
            trace["rounds"].append({"subquery": subq, "threshold": self.cfg.relevance_threshold, "n_hits": len(hits)})

        selected = self.deduplicate_and_diversify(all_sources, final_top_k=self.cfg.final_top_k)
        sufficiency = self.evaluate_sufficiency(plan, selected)
        trace["initial_sufficiency"] = asdict(sufficiency)

        if not sufficiency.is_sufficient:
            retry_sources: list[dict[str, Any]] = []
            for subq in plan.subqueries:
                hits = self.retrieve_one_subquery(
                    subquery=subq,
                    plan=plan,
                    threshold=self.cfg.retry_relevance_threshold,
                    vector_top_k=max(self.cfg.vector_top_k, 50),
                    bm25_top_k=max(self.cfg.bm25_top_k, 50),
                    candidate_top_k=max(self.cfg.candidate_top_k, 50),
                    rerank_top_k=max(self.cfg.rerank_top_k, self.cfg.retry_final_top_k, 12),
                )
                retry_sources.extend(hits)
                trace["rounds"].append({"subquery": subq, "threshold": self.cfg.retry_relevance_threshold, "n_hits": len(hits), "retry": True})

            combined = all_sources + retry_sources
            selected = self.deduplicate_and_diversify(combined, final_top_k=self.cfg.retry_final_top_k)
            sufficiency = self.evaluate_sufficiency(plan, selected)
            trace["retry_sufficiency"] = asdict(sufficiency)

        status = "ok" if selected and sufficiency.max_score >= self.cfg.retry_relevance_threshold else "low_confidence"
        message = "Знайдено релевантні джерела." if status == "ok" else "У корпусі не знайдено достатньо релевантних фрагментів для надійної відповіді."

        return {
            "status": status,
            "message": message,
            "query": question,
            "plan": asdict(plan),
            "results": selected,
            "sufficiency": asdict(sufficiency),
            "agent_trace": trace,
        }

    def build_context(self, retrieval_output: dict[str, Any]) -> str:
        if retrieval_output.get("status") != "ok":
            return ""
        blocks: list[str] = []
        total_chars = 0
        for i, res in enumerate(retrieval_output.get("results", []), start=1):
            speakers = res.get("speakers") or []
            speakers_str = ", ".join(speakers) if speakers else "невідомо"
            stephanus = res.get("stephanus_refs") or []
            stephanus_str = ", ".join(stephanus) if stephanus else "немає"
            block = (
                f"[{i}]\n"
                f"Діалог: {res.get('dialogue')}\n"
                f"Сторінки PDF: {res.get('page_start')}-{res.get('page_end')}\n"
                f"Мовці: {speakers_str}\n"
                f"Stephanus-посилання: {stephanus_str}\n"
                f"Chunk ID: {res.get('chunk_id')}\n"
                f"Rerank score: {float(res.get('rerank_score', 0.0)):.4f}\n"
                f"Subquery: {res.get('subquery', '')}\n"
                f"Текст:\n{res.get('text', '')}\n"
            )
            if total_chars + len(block) > self.cfg.max_context_chars:
                break
            blocks.append(block)
            total_chars += len(block)
        return "\n\n".join(blocks)

    def format_sources(self, sources: list[dict[str, Any]]) -> str:
        if not sources:
            return "Джерела: немає."
        lines = ["Джерела:"]
        for i, s in enumerate(sources, start=1):
            lines.append(
                f"{i}. {s.get('dialogue')}, сторінки PDF {s.get('page_start')}-{s.get('page_end')}, "
                f"chunk_id: {s.get('chunk_id')}, rerank_score={float(s.get('rerank_score', 0.0)):.3f}"
            )
        return "\n".join(lines)

    def build_final_messages(self, question: str, context: str, retrieval_output: dict[str, Any], mode: str) -> list[dict[str, str]]:
        mode_instruction = ANSWER_MODES.get(mode, ANSWER_MODES["academic"])
        suff = retrieval_output.get("sufficiency", {})
        source_audit = (
            f"Кількість джерел: {suff.get('n_sources')}\n"
            f"Максимальний rerank_score: {suff.get('max_score')}\n"
            f"Середній rerank_score: {suff.get('mean_score')}\n"
            f"Кількість діалогів у джерелах: {suff.get('n_dialogues')}\n"
            f"Попередження: {', '.join(suff.get('reasons') or []) if suff.get('reasons') else 'немає'}"
        )

        system_prompt = (
            "Ти — українськомовний науковий асистент для роботи з корпусом діалогів Платона. "
            "Ти відповідаєш тільки на основі наданого CONTEXT. Не використовуй зовнішні знання, "
            "не вигадуй цитат, сторінок, назв діалогів або chunk_id. Якщо відповідь у CONTEXT не підтверджена, "
            "прямо скажи: 'У наданих фрагментах корпусу це прямо не зазначено.'"
        )
        user_prompt = f"""
РЕЖИМ ВІДПОВІДІ:
{mode_instruction}

SOURCE AUDIT:
{source_audit}

CONTEXT:
{context}

ПИТАННЯ:
{question}

ВИМОГИ ДО ВІДПОВІДІ:
1. Відповідай українською.
2. Спирайся тільки на CONTEXT.
3. Якщо робиш висновок понад прямий переказ, познач його як "Можлива інтерпретація".
4. Наприкінці обов'язково додай розділ "Джерела:" з назвами діалогів, сторінками PDF і chunk_id.
""".strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def ensure_sources_in_answer(self, answer: str, sources: list[dict[str, Any]]) -> str:
        answer = (answer or "").strip()
        lower = answer.lower()
        has_sources = "джерела" in lower and ("chunk_id" in lower or "plato_" in lower)
        if has_sources:
            return answer
        return answer + "\n\n" + self.format_sources(sources) if answer else self.format_sources(sources)

    def answer(
        self,
        question: str,
        mode: str = "academic",
        model: Optional[str] = None,
        log: bool = True,
    ) -> dict[str, Any]:
        start_time = time.time()
        retrieval_output = self.agentic_retrieve(question)

        if retrieval_output["status"] != "ok":
            answer_text = retrieval_output["message"]
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "mode": mode,
                "model": normalize_model_name(model or self.cfg.openrouter_model),
                "status": "low_confidence",
                "answer": answer_text,
                "retrieval": retrieval_output,
                "elapsed_sec": round(time.time() - start_time, 3),
            }
            if log:
                append_jsonl(self.log_path, record)
            return record

        context = self.build_context(retrieval_output)
        messages = self.build_final_messages(question, context, retrieval_output, mode)
        raw_answer = self.generator.chat(messages, model=model)
        answer_text = self.ensure_sources_in_answer(raw_answer, retrieval_output.get("results", []))

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "mode": mode,
            "model": normalize_model_name(model or self.cfg.openrouter_model),
            "status": "ok",
            "answer": answer_text,
            "sources": [
                {
                    "chunk_id": s.get("chunk_id"),
                    "dialogue": s.get("dialogue"),
                    "page_start": s.get("page_start"),
                    "page_end": s.get("page_end"),
                    "rerank_score": s.get("rerank_score"),
                    "hybrid_score": s.get("hybrid_score"),
                    "subquery": s.get("subquery"),
                }
                for s in retrieval_output.get("results", [])
            ],
            "retrieval": retrieval_output if self.cfg.save_agent_trace else {k: v for k, v in retrieval_output.items() if k != "agent_trace"},
            "context_chars": len(context),
            "elapsed_sec": round(time.time() - start_time, 3),
        }
        if log:
            append_jsonl(self.log_path, record)
        return record


if __name__ == "__main__":
    cfg = AgenticRAGConfig(
        index_dir="plato_vector_index_bge_m3",
        openrouter_model="google/gemma-4-31b-it:free",
    )
    engine = PlatoAgenticRAGEngine(cfg)
    result = engine.answer("Чому Сократ не боїться смерті?", mode="academic")
    print(result["answer"])
