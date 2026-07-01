"""
Dataset Builder — извлекает технические вопросы из Telegram чата ALD Pro.

Использование:
  python build_dataset.py                              # извлечь вопросы → dataset_raw.jsonl
  python build_dataset.py --dcd                        # + DCD классификация
  python build_dataset.py --stats                      # статистика по датасету
  python build_dataset.py --sample 5                    # показать 5 примеров
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter
from typing import Any

# ── Путь к файлу ──
CHAT_PATH = os.path.expanduser(
    r"~\Downloads\Telegram Desktop\ChatExport_2026-07-02\result.json"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Фильтры ──
QUESTION_PATTERNS = re.compile(
    r'(как\s+\w+|почему|зачем|что\s+делать|'
    r'может\s+ли|возможно\s+ли|есть\s+ли|подскажите|'
    r'помогите|кто-нибудь|кто\s+сталкивался|'
    r'есть\s+инструкция|как\s+настроить|как\s+сделать|'
    r'какой\s+(верси[еюя]|дистрибутив)|'
    r'какие\s+(настройки|параметры|особенности)|'
    r'почему\s+(не\s+работает|ошибка|вылетает)|'
    r'ошибка|проблема|не\s+работает|'
    r'как\s+правильно|стоит\s+ли|нужна\s+помощь)',
    re.IGNORECASE,
)

TECH_TERMS = re.compile(
    r'\b(sssd|krb5|ldap|dns|dhcp|samba|freeipa|'
    r'kerberos|nfs|smb|cifs|ldif|rpc|gpo|gpresult|'
    r'powershell|wmi|winrm|rdp|xfreerdp|vdi|vnc|'
    r'pxe|tftp|iscsi|bonding|vlan|bridge|nat|'
    r'firewall|iptables|nftables|selinux|apparmor|'
    r'postfix|dovecot|clamav|spamassassin|opendkim|'
    r'mysql|postgresql|redis|nginx|apache|php|'
    r'terraform|ansible|docker|k8s|kubernetes|puppet|'
    r'gitlab|jenkins|nexus|artifactory|'
    r'ald\s*pro|aldpro|astralinux|astra\s*linux|'
    r'workspad|termidesk|rupost|rusbitech|'
    r'vmware|proxmox|hyper-v|qemu|kvm|xen|'
    r'debian|ubuntu|centos|rhel|альт|alteros)',
    re.IGNORECASE,
)

SHORT_MSG_PATTERN = re.compile(r'^.{,15}$')
LINK_PATTERN = re.compile(r'https?://\S+')
FORWARD_PATTERN = re.compile(r'^\[.*?\]')
REPLY_PATTERN = re.compile(r'^@\w+')


def is_technical_question(msg: dict) -> bool:
    """Проверка, является ли сообщение техническим вопросом."""
    if msg.get("type") != "message":
        return False
    
    text = msg.get("text", "")
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Слишком короткие — пропускаем
    if len(text) < 20:
        return False
    
    # Ссылки одни — пропускаем
    if LINK_PATTERN.fullmatch(text):
        return False
    
    # Пересланные сообщения — пропускаем
    if text.startswith("[") and "]" in text[:20]:
        return False
    
    # Должен содержать вопрос ИЛИ технический термин
    has_question = QUESTION_PATTERNS.search(text) or text.rstrip().endswith("?")
    has_tech = TECH_TERMS.search(text)
    
    # Технический вопрос или описание проблемы
    if has_question and has_tech:
        return True
    
    # Даже без вопроса, если описана конкретная проблема с tech-термином
    if has_tech and any(w in text.lower() for w in [
        "ошибк", "проблем", "не работ", "падае", "зависа",
        "настройк", "конфиг", "установк", "миграци",
        "обновлени", "совместим", "поддержк",
    ]):
        return True
    
    return False


def extract_questions(
    chat_path: str = CHAT_PATH,
    max_messages: int = 2000,
    min_length: int = 20,
    max_length: int = 1500,
) -> list[dict]:
    """Извлечь технические вопросы из чата."""
    _ensure_dir()
    
    # Определяем размер файла для прогресса
    file_size = os.path.getsize(chat_path)
    
    with open(chat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    messages = data.get("messages", [])
    print(f"Загружено сообщений: {len(messages)}")
    
    questions = []
    seen_texts = set()
    
    for i, msg in enumerate(messages):
        if len(questions) >= max_messages:
            break
        
        text = msg.get("text", "")
        if not isinstance(text, str):
            continue
        
        text = text.strip()
        if len(text) < min_length or len(text) > max_length:
            continue
        
        # Дедикация по хешу
        text_hash = hashlib.md5(text[:100].encode()).hexdigest()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        
        if not is_technical_question(msg):
            continue
        
        questions.append({
            "id": msg.get("id", i),
            "actor": msg.get("actor", "?"),
            "date": msg.get("date", ""),
            "text": text,
            "length": len(text),
            "has_question": bool(QUESTION_PATTERNS.search(text)),
            "has_tech_term": bool(TECH_TERMS.search(text)),
        })
        
        if (i + 1) % 5000 == 0:
            print(f"  Прогресс: {i+1}/{len(messages)}, найдено: {len(questions)}")
    
    print(f"Извлечено вопросов: {len(questions)}")
    return questions


def add_dcd_classification(questions: list[dict]) -> list[dict]:
    """Добавить DCD классификацию к каждому вопросу."""
    sys.path.insert(0, os.path.dirname(__file__))
    from dcd_router import classify
    
    for q in questions:
        r = classify(q["text"])
        q["dcd_domain"] = r.get("domain", "")
        q["dcd_collection"] = r.get("collection", "")
        q["dcd_confidence"] = r.get("confidence", 0)
    
    return questions


# ── Сохранение ──

def save_dataset(questions: list[dict], name: str = "dataset"):
    """Сохранить датасет в разных форматах."""
    _ensure_dir()
    
    # JSONL
    path_jsonl = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
    with open(path_jsonl, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Сохранено: {path_jsonl} ({len(questions)} записей)")
    
    # CSV (только id, text, dcd_domain)
    path_csv = os.path.join(OUTPUT_DIR, f"{name}.csv")
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("id\ttext\tdcd_domain\tdcd_collection\tlabel\n")
        for q in questions:
            text_escaped = q["text"].replace("\t", " ").replace("\n", " ")
            f.write(f'{q["id"]}\t{text_escaped}\t'
                    f'{q.get("dcd_domain","")}\t'
                    f'{q.get("dcd_collection","")}\t\n')
    print(f"Сохранено: {path_csv}")
    
    # TXT (только тексты, 1 на строку — для LLM)
    path_txt = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q["text"] + "\n---\n")
    print(f"Сохранено: {path_txt}")


def print_stats(questions: list[dict]):
    """Статистика по датасету."""
    print(f"\n{'='*50}")
    print(f"СТАТИСТИКА ДАТАСЕТА")
    print(f"{'='*50}")
    print(f"Всего записей: {len(questions)}")
    print(f"Средняя длина: {sum(q['length'] for q in questions)/len(questions):.0f} символов")
    print(f"С вопросом:    {sum(1 for q in questions if q.get('has_question'))}")
    print(f"С tech-термом: {sum(1 for q in questions if q.get('has_tech_term'))}")
    
    if questions[0].get("dcd_domain"):
        domains = Counter(q["dcd_domain"] for q in questions)
        print(f"\nDCD домены:")
        for dom, cnt in domains.most_common(10):
            print(f"  {dom:25s}: {cnt:4d} ({cnt/len(questions)*100:5.1f}%)")


def show_sample(questions: list[dict], n: int = 5):
    """Показать примеры."""
    print(f"\n{'='*50}")
    print(f"ПРИМЕРЫ ВОПРОСОВ")
    print(f"{'='*50}")
    for q in questions[:n]:
        dcd = q.get("dcd_domain", "?")
        print(f"\n[{q['id']}] ({dcd})")
        print(f"  {q['text'][:300]}")


# ── Main ──

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcd", action="store_true", help="Добавить DCD классификацию")
    parser.add_argument("--stats", action="store_true", help="Статистика")
    parser.add_argument("--sample", type=int, default=0, help="Показать N примеров")
    parser.add_argument("--max", type=int, default=2000, help="Максимум вопросов")
    args = parser.parse_args()
    
    # 1. Извлечь вопросы
    questions = extract_questions(max_messages=args.max)
    
    if not questions:
        print("Нет данных для обработки")
        return
    
    # 2. DCD
    if args.dcd or args.stats:
        print("\nДобавляю DCD классификацию...")
        questions = add_dcd_classification(questions)
    
    # 3. Сохранить
    save_dataset(questions)
    
    # 4. Статистика
    if args.stats:
        print_stats(questions)
    
    # 5. Примеры
    if args.sample > 0:
        show_sample(questions, args.sample)


if __name__ == "__main__":
    main()