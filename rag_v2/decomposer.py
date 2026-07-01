"""
Query Decomposer — LLM разбивает запрос на подзапросы по источникам.

Стратегия:
  1. Определить сущности (SSSD, krb5, ALD Pro, MSAD)
  2. Определить тип каждой сущности:
     - product → Confluence, Lodestone
     - config/parameter → wiki, general Linux docs
     - bug/incident → Jira
     - code/lib → Context7
  3. Сгенерировать подзапрос (каждый в свой источник)
  4. LLM решает какие подзапросы нужны и куда
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rag_config import LM_STUDIO_CHAT_URL

_LLM_URL = LM_STUDIO_CHAT_URL.rstrip('/')

import aiohttp

DECOMPOSER_PROMPT = """Ты — система анализа запросов для RAG. Разбей запрос пользователя на подзапросы.

Правила:
- Каждый подзапрос идёт в конкретный источник
- Не более 4 подзапросов
- source: одно из "confluence", "jira", "lodestone", "context7", "zvec", "web"
- Если запрос про продукт/настройку — confluence или lodestone
- Если про баг/инцидент/тикет — jira
- Если про код/библиотеку/технологию — context7
- Если общая Linux настройка — zvec+web
- Если запрос составной (продукт + технология) — несколько подзапросов

Ответь ТОЛЬКО JSON в формате:
{subqueries_json}

Запрос: {query}"""


async def decompose(query: str, session: aiohttp.ClientSession | None = None) -> list[dict]:
    """Разбить запрос на подзапросы. Возвращает список [{{query, source, reason}}]."""
    SQA = '{{"subqueries": [{{"query": "..."}}]}}'
    prompt = DECOMPOSER_PROMPT.format(query=query, subqueries_json=SQA)
    
    payload = {
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    
    async def _call(s: aiohttp.ClientSession):
        async with s.post(_LLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            # Извлечь JSON из ответа (может быть в ```json ... ```)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
            return result.get("subqueries", [])
    
    if session:
        return await _call(session)
    
    async with aiohttp.ClientSession() as s:
        return await _call(s)


def extract_entities(query: str) -> list[str]:
    """Извлечь ключевые сущности из запроса (быстро, без LLM)."""
    import re
    # Технические термины (слова с цифрами, версиями, дефисами)
    terms = re.findall(r'\b[A-Za-z][A-Za-z0-9_.\-/]{2,}\b', query)
    # Русские термины (длинные слова)
    ru_terms = re.findall(r'\b[А-Яа-я][А-Яа-яё]{3,}\b', query)
    return list(set(terms + ru_terms))