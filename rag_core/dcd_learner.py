"""
DCD Learner — обучает DCD роутер на основе реальных результатов поиска.

Принцип:
  1. Читаем кеш (или лог) запросов
  2. Для каждого: DCD отправил в domainX, но реальный контент нашёлся в domainY
  3. Извлекаем ключевые слова из запроса
  4. Предлагаем патчи для dcd_router.py:
     - Добавить ключевые слова в domainY (правильный)
     - Добавить anti_keywords в domainX (неправильный)
  5. Авто-патч если уверенность > 0.8

Запуск:
  python dcd_learner.py            # анализ + предложение
  python dcd_learner.py --apply    # анализ + авто-применение
  python dcd_learner.py --log      # просто показать лог
"""

import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any

# ── DCD domain → source mapping ──
# Какие источники ожидаются для каких DCD доменов
DOMAIN_EXPECTED_SOURCES = {
    "rusbitech": {"lodestone", "confluence", "jira"},
    "devops": {"context7"},
    "software-dev": {"context7"},
    "security": {"jira", "context7"},
    "automation": {"context7"},
}

# Обратное: source → ожидаемый domain
SOURCE_EXPECTED_DOMAINS = {
    "lodestone": "rusbitech",
    "confluence": "rusbitech",
    "jira": "rusbitech",
    "context7": "devops",  # software-dev тоже сюда
    "zvec": "research",
}

# Домены, для которых context7 — правильный источник
_CONTEXT7_DOMAINS = {"devops", "software-dev", "automation"}


def read_cache() -> list[dict]:
    """Читаем in-memory кеш (через импорт модуля)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from rag_async import _CACHE
    records = []
    for key, value in list(_CACHE.items()):
        dcd_domain = value.get("dcd_domain", value.get("_trace", {}).get("domain", ""))
        dcd_collection = value.get("dcd_collection", "")
        source = value.get("source", "?")
        chunks = value.get("chunks", [])
        has_content = len(chunks) > 0
        records.append({
            "key": key,
            "dcd_domain": dcd_domain,
            "dcd_collection": dcd_collection,
            "source": source,
            "has_content": has_content,
            "chunks_count": len(chunks),
        })
    return records


def read_routing_log() -> list[dict]:
    """Читаем persistent лог (если есть)."""
    log_path = os.path.join(os.path.dirname(__file__), "routing_log.jsonl")
    records = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def log_routing(query: str, dcd_result: dict, rag_result: dict):
    """Записать один запрос в лог (вызывается из pipeline)."""
    log_path = os.path.join(os.path.dirname(__file__), "routing_log.jsonl")
    entry = {
        "query": query[:200],
        "dcd_domain": dcd_result.get("domain", ""),
        "dcd_collection": dcd_result.get("collection", ""),
        "dcd_confidence": dcd_result.get("confidence", 0),
        "actual_source": rag_result.get("source", "?"),
        "has_content": len(rag_result.get("chunks", [])) > 0,
        "chunks_count": len(rag_result.get("chunks", [])),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_keywords(query: str, max_keywords: int = 5) -> list[str]:
    """Извлечь ключевые слова из запроса (слова, не цифры)."""
    # Технические термины
    terms = re.findall(r'\b[A-Za-z][A-Za-z0-9_.-]{2,}\b', query)
    # Русские термины (длиннее 3 букв)
    ru_terms = re.findall(r'\b[А-Яа-яё][А-Яа-яё]{3,}\b', query)
    # Объединить, убрать короткие/шумовые
    stop_words = {"the", "and", "for", "not", "how", "why", "what", "its",
                  "set", "use", "add", "get", "run", "new", "old",
                  "настройка", "установка", "конфигурация", "параметр",
                  "linux", "windows", "debian", "ubuntu"}
    combined = []
    for w in terms + ru_terms:
        wl = w.lower()
        if len(w) >= 3 and wl not in stop_words:
            combined.append(w)
    return combined[:max_keywords]


def analyze(misrouted: list[dict]) -> list[dict]:
    """Анализ неверно направленных запросов. Возвращает предложения патчей."""
    suggestions = []
    
    # Группируем по паре (wrong_domain → correct_domain)
    pairs = defaultdict(list)
    for rec in misrouted:
        wrong = rec["dcd_domain"]
        correct = rec["actual_source_domain"]
        keywords = extract_keywords(rec.get("query", ""))
        pairs[(wrong, correct)].append({
            "query": rec["query"],
            "keywords": keywords,
            "source": rec.get("actual_source", "?"),
        })
    
    for (wrong, correct), examples in pairs.items():
        # Собираем частотность ключевых слов
        kw_counts = Counter()
        for ex in examples:
            for kw in ex["keywords"]:
                kw_counts[kw] += 1
        
        # Топ-10 ключевых слов
        top_kw = [kw for kw, _ in kw_counts.most_common(10)]
        
        # Собираем примеры запросов
        sample_queries = [ex["query"][:80] for ex in examples[:5]]
        
        suggestions.append({
            "type": "add_keywords",
            "target_domain": correct,
            "keywords": top_kw,
            "sample_queries": sample_queries,
            "confidence": len(examples) / max(len(misrouted), 1),
            "sql": f"INSERT INTO domain_keywords (domain, keyword, weight) "
                   f"VALUES ('{correct}', '{top_kw[0] if top_kw else ''}', 3)",
            "_readable": f"Добавить {top_kw[:5]} в {correct} (примеры: {sample_queries[0][:40]}...)"
        })
        
        suggestions.append({
            "type": "add_anti_keywords",
            "target_domain": wrong,
            "keywords": top_kw,
            "sample_queries": sample_queries,
            "confidence": len(examples) / max(len(misrouted), 1),
            "sql": f"INSERT INTO domain_anti_keywords (domain, keyword) "
                   f"VALUES ('{wrong}', '{top_kw[0] if top_kw else ''}')",
            "_readable": f"Добавить anti_keywords {top_kw[:3]} к {wrong}"
        })
    
    return suggestions


def patch_dcd(suggestions: list[dict], dry_run: bool = True):
    """Применить предложения к dcd_router.py."""
    dcd_path = os.path.join(os.path.dirname(__file__), "dcd_router.py")
    with open(dcd_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    patches = []
    for s in suggestions:
        if s["type"] == "add_keywords":
            domain = s["target_domain"]
            keywords = [kw.lower() for kw in s["keywords"] if len(kw) >= 3]
            if not keywords:
                continue
            
            # Находим секцию keywords для этого domain
            # Формат: "domain": {"keywords": {"kw": weight, ...}, "anti_keywords": [...]}
            pattern = rf'"{domain}":\s*{{\s*"keywords":\s*{{'
            match = re.search(pattern, content)
            if match:
                # Добавляем новые ключевые слова
                insert_pos = match.end()
                additions = []
                for kw in keywords[:5]:
                    additions.append(f'"{kw}": 3, ')
                new_content = content[:insert_pos] + "\n" + "".join(additions) + content[insert_pos:]
                patches.append((dcd_path, new_content))
                if not dry_run:
                    content = new_content
        
        elif s["type"] == "add_anti_keywords":
            domain = s["target_domain"]
            keywords = [kw.lower() for kw in s["keywords"] if len(kw) >= 3]
            if not keywords:
                continue
            
            # Находим секцию anti_keywords
            pattern = rf'"{domain}":\s*{{[^}}]*"anti_keywords":\s*\['
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                # anti_keywords нет — создаём
                pattern = rf'"{domain}":\s*{{\s*"keywords":\s*{{[^}}]+\}}'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    insert_pos = match.end()
                    anti = ',\n      "anti_keywords": ["' + '", "'.join(keywords[:3]) + '"]\n    }'
                    new_content = content[:match.end()-1] + anti + content[match.end()-1:]
                    patches.append((dcd_path, new_content))
                    if not dry_run:
                        content = new_content
            else:
                # Добавляем к существующим anti_keywords
                insert_pos = match.end() - 1
                additions = ', "' + '", "'.join(keywords[:3]) + '"'
                new_content = content[:insert_pos] + additions + content[insert_pos:]
                patches.append((dcd_path, new_content))
                if not dry_run:
                    content = new_content
    
    # Применяем патчи
    if not dry_run:
        for path, new_content in patches:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
    
    return patches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Auto-apply patches")
    parser.add_argument("--log", action="store_true", help="Show routing log")
    args = parser.parse_args()
    
    # 1. Собираем данные
    cache = read_cache()
    log = read_routing_log()
    records = log + cache
    
    if args.log:
        print(f"Routing log: {len(log)} entries")
        print(f"Cache: {len(cache)} entries")
        print()
        for rec in records[-20:]:
            print(f'{rec.get("dcd_domain","?"):15s} → {rec.get("source","?"):12s} '
                  f'chunks={rec.get("chunks_count","?"):2d}  {rec.get("query","")[:50]}')
        return
    
    if not records:
        print("Нет данных. Запустите несколько запросов через RAG сначала.")
        return
    
    # 2. Находим неверно направленные
    misrouted = []
    for rec in records:
        dcd = rec.get("dcd_domain", "")
        src = rec.get("source", "")
        has_content = rec.get("has_content", True)
        
        expected = SOURCE_EXPECTED_DOMAINS.get(src, "")
        if expected and dcd != expected and dcd:
            # context7 подходит для software-dev, devops, automation
            if src == "context7" and dcd in _CONTEXT7_DOMAINS:
                continue
            # zvec — fallback для всех доменов (не ошибка DCD, а ошибка контента)
            if src == "zvec":
                continue
            rec["actual_source_domain"] = expected
            misrouted.append(rec)
    
    print(f"Всего записей: {len(records)}")
    print(f"Неверно направлены: {len(misrouted)}")
    print()
    
    if not misrouted:
        print("Все маршрутизированы правильно!")
        return
    
    # 3. Статистика
    pairs = Counter()
    for rec in misrouted:
        pair = f'{rec["dcd_domain"]} → {rec["actual_source_domain"]}'
        pairs[pair] += 1
    
    print("Систематические ошибки:")
    for pair, count in pairs.most_common():
        print(f"  {pair}: {count}")
    print()
    
    # 4. Предложения
    suggestions = analyze(misrouted)
    
    print("Предлагаемые патчи:")
    for s in suggestions:
        confidence = s.get("confidence", 0)
        marker = "✓" if confidence > 0.5 else "?"
        print(f'  {marker} {s["_readable"][:80]} (conf={confidence:.0%})')
    
    if args.apply:
        patches = patch_dcd(suggestions, dry_run=False)
        print(f"\nПрименено {len(patches)} патчей к dcd_router.py")
    else:
        print(f"\nЗапусти с --apply для применения")
    
    # 5. Если лога нет — предлагаем инициализировать
    if not log:
        print("\nСовет: добавь log_routing() вызов в rag_async.py после каждого поиска")


if __name__ == "__main__":
    main()