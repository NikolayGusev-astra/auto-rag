"""DCD Router —  keyword-matching классификатор запроса по домену и коллекции.

Назначение: Определить домен и коллекцию по пользовательскому запросу.
Тип: Негенеративный (keyword matching + эвристики).
Latency: <50ms на 13 доменах.
Зависимости: Только stdlib (re, json, pathlib). Нет LLM-вызовов.

Формат выхода:
{
  "domain": "software-dev",
  "collection": "code-review",
  "confidence": 0.92,
  "keywords_matched": ["код", "ревью", "PR"],
  "fallback": false
}
"""

import re
import json
from pathlib import Path

# ─── Конфигурация доменов ───────────────────────────────────────────

# Веса: 3 = сильный триггер (специфичный для домена)
#        2 = средний (часто встречается в домене)
#        1 = слабый (может пересекаться с другими доменами)
DOMAIN_KEYWORDS = {
    "rusbitech": {
        "weight": 3,
        "anti_keywords": ["nginx", "docker", "letsencrypt", "pytest", "async function",
                          "javascript", "react", "typescript", "git", "github",
                          "ci/cd", "redis", "postgresql", "kubernetes",
                          "cve", "patroni", "etcd", "aes", "gost", "шифрование",
                          "инцидент", "инцидента", "adr ", "sbl", "sbl "],
        "keywords": {
            # Продукты
            "ald pro": 7, "aldpro": 7, "ald-pro": 7,
            "acm": 5, "astra control manager": 6, "msad": 5,
            "rupost": 5, "ru-post": 5, "rupost": 5,
            "alse": 5, "astra linux special edition": 5,
            "astra linux": 4, "astralinux": 4,
            "ald": 4, "astra linux domain": 5, "aldpro": 7,
            "keycloak": 4,
            "lodestone": 4,
            "protopack": 4,
            # Заказчики
            "русгидро": 5, "rusgidro": 5, "rus hydro": 5,
            "татнефть": 5, "tatneft": 5,
            "новатэк": 5, "novatek": 5, "НОВАТЭК": 5,
            "газпром": 5, "gazprom": 5,
            "россети": 4, "roseti": 4,
            "алабуга": 4, "alabuga": 4,
            "увз": 3,
            # Пресейл
            "presale": 3, "пресейл": 3, "presale-": 4,
            "пми": 3, "пси": 3, "пилотирование": 3,
            # Типы работ
            "rca": 4, "root cause": 4, "root-cause": 4,
            "hld": 3, "lld": 3, "типовой проект": 3,
            "стендирование": 3, "внедрение": 3, "домен": 2,
            "ддо": 5, "кд": 3, "контроллер домена": 6,
            "аргументация": 2, "аудит": 2, "сайзинг": 3,
            "vulnerability": 2, "уязвимость": 2, "cve": 3,
            "samba": 4, "aldprosam": 6, "libipa-aldpro": 6, "доверитель": 5,
            "smart-card": 2, "смарт-карта": 2, "токен": 2,
            "rodc": 3, "родс": 3,
            # Дополнительные для маршрутизации
            "ретроспектива": 3,    # для РусГидро
            "workspad": 4, "workspad x": 4, "works pad": 3,
            # Общее
            "astra": 2, "astralinux.ru": 3, "rusbitech": 3,
            "jira": 1, "tempo": 1,
        },
        "collections": {
            "rusbitech-products": [
                "ald pro", "aldpro", "acm", "rupost", "alse", "keycloak",
                "lodestone", "protopack", "workspad",
                "почтовый сервер", "почта", "postfix", "dovecot",
                "сертификат", "pki", "центр сертификации",
                "мобильное рабочее место",
                "nova", "дистрибутив", "сертификация фичи",
                "sso", "oidc", "аутентификация",
                "антивирус", "антиспам", "фильтрация",
            ],
            "rusbitech-customers": [
                "русгидро", "татнефть", "новатэк", "газпром", "россети",
                "алабуга", "увз", "роснефть", "мвд",
            ],
            "rusbitech-deployment": [
                "стендирование", "внедрение", "деплой", "развертывание", "пилотирование",
                "настройка", "установка", "миграция", "обновление",
                "смена ip", "смена", "конфигурация", "развертывания",
                "автоматизация", "playbook", "ansible", "terraform",
                "vm", "автоматизация развертывания",
                "переход", "перенос",
            ],
            "rusbitech-architecture": [
                "hld", "lld", "архитектура", "типовой проект", "сайзинг",
                "adr", "проектирование", "схема", "диаграмма",
                "архитектурное решение",
            ],
            "rusbitech-rca": [
                "rca", "root cause", "анализ причин", "расследование",
                "инцидент", "падение сервера", "падение",
                "отказ", "сбой", "анализ",
            ],
            "rusbitech-vulnerability": [
                "cve", "уязвимость", "vulnerability",
                "патч", "обновление безопасности",
            ],
            "rusbitech-security": [
                "мандатный доступ", "sbl", "замкнутая среда",
                "сертификация безопасности", "сертификация",
                "аудит безопасности", "аудит",
                "шифрование", "aes", "gost", "gcm", "защита",
            ],
            "rusbitech-presale": [
                "presale", "пресейл", "консультация", "импортозамещение",
                "анализ проекта", "пми", "пси",
                "консультация пресейл",
            ],
            "rusbitech-jira": [
                "presale", "пресейл", "PRESALE-", "jira",
                "татнефть", "tatneft", "новатэк", "novatek", "НОВАТЭК",
                "газпром", "gazprom", "алабуга", "alabuga",
                "тикет", "issue", "backlog",
            ],
        },
    },
    "software-dev": {
        "weight": 3,
        "anti_keywords": ["лог", "сервер", "деплой", "docker", "nginx",
                          "send message", "отправь", "cron", "бэкап",
                          "комикс", "инфографика", "видео-анимация", "песня", "музыка",
                          "нарисуй", "пиксель-арт", "ascii art", "ascii баннер"],
        "keywords": {
            # Русские
            "код": 3, "код-ревью": 3, "ревью": 3, "pr": 3, "pull request": 3,
            "дебаг": 3, "отладка": 3, "debug": 3, "задебажь": 3,
            "клиент": 2, "пакет": 2, "force": 2, "sudo": 2,
            "архитектура": 3, "архитектурная диаграмма": 5, "архитектурную диаграмму": 5,
            "рефакторинг": 3, "spike": 3,
            "тест": 3, "тесты": 3, "tdd": 3, "тестирование": 3,
            "планирование": 3, "plan": 3, "декомпозиция": 2,
            "simplify": 3, "simplify-code": 3,
            "питон": 2, "python": 2, "js": 1, "javascript": 1, "rust": 2, "typescript": 1,
            "функция": 2, "функцию": 2, "класс": 2, "модуль": 2,
            "ошибка": 2, "баг": 2, "bug": 2, "exception": 2,
            "lint": 2, "линтер": 2, "formatting": 1,
            "ci/cd": 2, "unit test": 3, "integration test": 3,
            "e2e": 3, "test coverage": 3, "юнит-тест": 3,
            "adr": 3, "напиши adr": 3,
        },
        "collections": {
            "code-review": ["код-ревью", "code review", "ревью", "pr", "pull request",
                           "review pr", "проверь код", "check code"],
            "architecture": ["архитектура", "adr", "design", "диаграмма", "diagram",
                            "system design", "ipc pattern", "expansion point",
                            "excalidraw", "архитектурная диаграмма",
                            "создай архитектурную диаграмму",
                            "напиши adr", "adr для сервиса"],
            "debugging": ["дебаг", "debug", "ошибка", "bug", "traceback",
                         "exception", "stack trace", "отладка", "log", "лог",
                         "задебажь", "падение", "падения", "найди причину"],
            "testing": ["тест", "test", "tdd", "тестирование", "unit test",
                       "integration test", "e2e", "test coverage",
                       "юнит-тест", "запусти тесты", "найди причину падения"],
            "planning": ["план", "plan", "декомпозиция", "задача", "задачи",
                        "spike", "writing plan", "scope",
                        "adr", "напиши adr", "архитектурное решение",
                        "план миграции", "план миграции данных",
                        "создай план"],
        },
    },
    "devops": {
        "weight": 3,
        "keywords": {
            "деплой": 3, "deploy": 3, "сервер": 3, "хостинг": 3,
            "rag": 2, "индексация": 3, "zvec": 2, "embedding": 2, "вектор": 2,
            "docker": 3, "docker-compose": 3,
            "linux": 2, "ubuntu": 2, "debian": 2, "centos": 2,
            "nginx": 3, "apache": 2,
            "vpn": 3, "openvpn": 3, "xray": 3, "reality": 3,
            "zabbix": 3, "prometheus": 3, "grafana": 3, "мониторинг": 3, "мониторинга": 3, "метрики": 3, "healthcheck": 3,
            "security": 2, "sbl": 3, "аудит": 2,
            "ip": 1, "dns": 2, "ssl": 2, "tls": 2, "cert": 2, "tftp": 3, "kerberos": 2, "krb5": 2,
            "systemd": 3, "service": 2, "daemon": 2,
            "ufw": 3, "firewall": 3, "iptables": 3,
            "synapse": 3, "matrix": 3, "telegram": 2,
            "мониторинг для": 3, "healthcheck для": 3,
            "бекап базы": 3, "создай бекап": 3,
            # Astra Linux / ALD Pro specific
            "astra": 3, "astralinux": 3, "aldpro": 3, "ald pro": 3,
            "ддо": 3, "msad": 3, "samba": 3, "active directory": 3,
            "domain controller": 3, "ldap": 3, "kerberos": 3,
            "настройка ддо": 5, "настройка aldpro": 5, "настройка msad": 5,
            "ддо aldpro": 5, "msad aldpro": 5,
            "конфигурация ддо": 5, "конфигурация aldpro": 5,
        },
        "anti_keywords": ["сравни", "сравнение", "vs", "проаналижи лог", "лог ошибок", "научная", "научный",
                          "firewall правила", "настрой firewall", "ufw правила", "iptables правила",
                          "vulnerability", "cve", "exploit", "threat model", "red team",
                          "jailbreak", "injection", "xss", "sql injection"],
        "collections": {
            "rag-deploy": ["zvec", "chroma", "индексация", "эмбеддинг",
                          "векторный поиск", "переиндексировать",
                          "rag поиск", "rag индекс", "поиск по wiki",
                          "задеплой rag", "задеплой rag", "rag на hq",
                          "оптимизируй скорость rag", "проанализируй производительность rag"],
            "deployment": ["деплой", "deploy", "docker", "docker-compose",
                          "systemd", "service", "daemon", "бэкап", "backup",
                          "restore", "бекап базы", "создай бекап"],
            "infrastructure": ["сервер", "linux", "nginx", "dns", "ssl",
                              "ip", "tcp", "network", "vpn", "openvpn",
                              "xray", "reality", "reverse proxy",
                              "ssl сертификат", "переключить домен"],
            "monitoring": ["мониторинг", "метрики", "healthcheck", "лог",
                          "log", "prometheus", "grafana", "alert", "дашборд",
                          "dashboard", "health", "status", "uptime",
                          "работоспособность", "создай дашборд",
                          "мониторинг для", "healthcheck для",
                          "уведомления о падении", "alert о падении",
                          "падение сервера", "сервер упал"],
            "security": ["security", "sbl", "аудит", "firewall", "ufw",
                         "iptables", "sanitize", "forensics"],
        },
    },
    "publishing": {
        "weight": 3,
        "anti_keywords": ["научная", "научный", "федерация", "rag",
                          "исследование", "arxiv", "academic",
                          "автоматическая публикация", "автопубликация",
                          "cron", "webhook", "бэкап"],
        "keywords": {
            "статья": 3, "article": 3, "publish": 3,
            "telegram": 2, "telegraph": 2, "телеграм": 2,
            "habr": 3, "хабр": 3, "medium": 3,
            "стиль": 2, "style": 2, "верификация": 2,
            "оформление": 2, "редактирование": 2, "proofread": 3,
            "posting": 3, "публикация": 3,
            "контент": 2, "content": 2, "cms": 2,
            "форматирование": 2, "markdown": 2, "rich message": 2,
            "оформи пост": 3, "напиши пост": 3, "пост для": 3,
        },
        "collections": {
            "articles": ["статья", "article", "habr", "хабр",
                        "medium", "черновик", "draft",
                        "напиши статью", "хочу написать статью",
                        "пост про"],
            "style_verification": ["стиль", "style", "верификация",
                                  "оформление", "proofread", "редактирование",
                                  "красивое форматирование"],
            "posting": ["telegram", "telegraph", "постинг", "публикация",
                       "rich message", "posting workflow",
                       "опубликуй пост", "пост в телеграм",
                       "напиши пост про", "оформи пост"],
        },
    },
    "research": {
        "weight": 2,
        "anti_keywords": ["сервер", "деплой", "docker", "nginx", "firewall",
                          "создать задача", "cron", "бэкап", "backup",
                          "линтер", "тест", "код-ревью", "refactor",
                          "сравни", "сравнение", "vs",
                          "lodestone", "confluence", "документация",
                          "техническая документация", "wiki страница"],
        "keywords": {
            "исследование": 3, "research": 3, "arxiv": 3, "paper": 3,
            "wiki": 3, "wiki-sync": 3, "llm wiki": 3,
            "поиск": 2, "search": 2, "найди": 2, "поисковик": 2,
            "блог": 2, "blog": 2, "blogwatcher": 3,
            "осинт": 3, "osint": 3, "расследование": 3,
            "анализ": 2, "analysis": 2, "проанализируй": 2,
            "polymarket": 3, "рынок": 2, "прогноз": 2,
            "comparison": 2, "сравни": 2, "vs": 2,
            "llm": 3, "large language model": 3, "модель": 2, "модели": 2,
            "claude": 3, "модель claude": 3, "новые модели": 3,
            "федерация": 3, "rag федерация": 3,
            "альтернативы": 3, "альтернатива": 3, "найди альтернативы": 3,
        },
        "collections": {
            "academic": ["arxiv", "paper", "research", "academic", "научный",
                        "научная статья", "статья", "федерация",
                        "rag федерация", "научную статью"],
            "wiki": ["wiki", "llm wiki", "wiki-sync", "knowledge graph",
                    "kb", "база знаний", "wiki страница", "обнови wiki"],
            "web_search": ["поиск", "search", "блог", "blog", "blogwatcher",
                          "google", "searx", "perplexity", "найди информацию",
                          "найди информацию про", "модель claude",
                          "альтернативы chromadb", "новые модели",
                          "модели llm", "исследуй модели"],
            "osint": ["osint", "расследование", "public records",
                      "пропublic"],
            "analysis": ["анализ", "analysis", "сравнение", "summary",
                        "polymarket", "market research",
                        "сравни", "подготовь отчет", "подготовь отчёт",
                        "проанализируй лог", "лог ошибок", "отчет по проекту",
                        "сравни zvec", "сравни chroma", "zvec vs chroma"],
        },
    },
    "automation": {
        "weight": 3,
        "anti_keywords": ["код-ревью", "bug", "ошибка в коде", "деплой кода",
                          "статья", "пост", "публикация", "wiki",
                          "архитектура", "excalidraw", "диаграмма",
                          "миграция", "migrate", "data migration",
                          "linear", "notion", "airtable",
                          "план", "plan", "создай план", "adr",
                          "бэкап базы", "backup базы", "restore базы",
                          "zvec", "chroma",
                          "nginx", "docker",
                          "лог ошибок", "падение сервера", "сервер упал",
                          "firewall", "vulnerability", "аудит безопасности",
                          "proxmox", "backup server", "pbs", "резервное копирование",
                          "debian", "vm", "virtual machine"],
        "keywords": {
            "cron": 3, "cronjob": 3, "автопланировщик": 3, "schedule": 3,
            "webhook": 3, "subscription": 2, "event-driven": 3,
            "скрипт": 2, "script": 2, "автоматизация": 3, "automation": 3,
            "kanban": 3, "linear": 2, "todoist": 2, "notion": 2,
            "пайплайн": 2, "pipeline": 2, "batch": 2,
            "парсер": 3, "scraper": 3, "parsing": 3,
            "notification": 3, "уведомление": 3, "alert": 2,
            "migration": 2, "migrate": 2, "data migration": 3,
            "parser": 3, "парсинг": 3, "миграция": 2,
            "cron задача": 3, "cron задачу": 3, "cron": 3,
            "задача для": 3, "задачу для": 3, "cron задача бэкап": 4,
            "бэкап": 3, "backup": 3, "restore": 3,
            "расписание": 3, "расписани": 3, "задача задача": 2,
            "автомат": 2, "триггер": 2, "event": 2,
            "поменять работу cron": 3, "cron задача": 3,
        },
        "collections": {
            "cron": ["cron", "cronjob", "schedule", "автопланировщик", "periodic"],
            "webhooks": ["webhook", "subscription", "event-driven",
                        "event subscription",
                        "автоматическая публикация", "автопубликация",
                        "публикация по расписанию", "cron публикация"],
            "kanban": ["kanban", "linear", "todoist", "notion", "task management",
                       "issue tracking"],
            "scripts": ["скрипт", "script", "парсер", "parser", "scraper",
                       "batch", "pipeline"],
            "notifications": ["notification", "уведомление", "alert",
                             "message", "оповещение",
                             "уведомления о падении", "alert о падении",
                             "настрой уведомления"],
        },
    },
    "data-ml": {
        "weight": 2,
        "anti_keywords": ["статья", "пост", "публикация", "код-ревью", "деплой",
                          "сервер", "firewall", "security audit", "отправь",
                          "send message", "wiki", "блог", "blogwatcher",
                          "diataxis", "prism", "clarity", "ponytail"],
        "keywords": {
            "ml": 3, "machine learning": 3, "mlops": 3,
            "llm": 3, "large language model": 3, "модель": 2, "модели": 2,
            "inference": 3, "инференс": 3, "serving": 2,
            "dspy": 3, "prompt": 2, "fine-tune": 3, "finetune": 3,
            "dataset": 2, "датасет": 2, "training": 2, "training run": 2,
            "benchmark": 3, "evaluation": 2, "eval": 2,
            "quantization": 3, "gguf": 3, "gguf": 3, "vllm": 3,
            "huggingface": 3, "hf": 2, "token": 1, "tokenization": 2,
            "segment": 2, "image": 2, "video": 2,
            "classification": 2, "ner": 2, "embedding": 2,
            "обучи": 3, "распознавание": 3, "изображение": 3, "изображений": 3,
        },
        "collections": {
            "mlops": ["mlops", "machine learning", "ml pipeline", "training",
                     "fine-tune", "finetune", "dataset",
                     "обучи модель", "распознавание изображений"],
            "inference": ["inference", "инференс", "serving", "vllm",
                         "llama.cpp", "gguf", "quantization", "benchmark"],
            "dspy": ["dspy", "prompt engineering", "prompt optimization",
                     "declarative program"],
            "multimodal": ["audio", "music", "image", "video", "segment",
                          "musicgen", "audiocraft", "stable diffusion",
                          "изображение", "изображений", "распознавание",
                          "сгенерируй музыку", "музыку для видео",
                          "видео с музыкой"],
        },
    },
    "integrations": {
        "weight": 2,
        "keywords": {
            "github": 3, "gh": 2, "gitlab": 3,
            "google": 2, "gws": 2, "gmail": 2, "calendar": 2, "drive": 2,
            "notion": 3, "airtable": 3, "linear": 2, "настрой linear": 3,
            "ozon": 3, "wildberries": 3, "wb": 2, "avito": 3,
            "feishu": 3, "lark": 2,
            "spotify": 3, "music": 2,
            "discord": 3, "slack": 2,
            "homeassistant": 3, "openhue": 3, "smart home": 3,
            "api": 1, "rest": 2, "graphql": 2,
            "integration": 2, "интеграция": 2,
        },
        "collections": {
            "github": ["github", "gh", "gitlab", "pull request", "issue",
                       "code review", "repository", "github actions",
                       "linear управления задачами", "настрой linear"],
            "google": ["google", "gws", "gmail", "calendar", "drive",
                      "google forms", "google workspace"],
            "productivity": ["notion", "airtable", "linear", "todoist",
                            "task management", "issue tracking",
                            "google docs", "google forms", "google workspace",
                            "google документ", "google документы",
                            "linear задача", "linear статус",
                            "обнови статус задачи в linear"],
            "commerce": ["ozon", "wildberries", "wb", "avito", "seller",
                        "marketplace", "авито интеграция"],
            "feishu": ["feishu", "lark", "doc", "drive"],
            "smart_home": ["homeassistant", "openhue", "smart home",
                          "light", "switch", "sensor"],
        },
    },
    "analysis": {
        "weight": 2,
        "anti_keywords": ["код", "ревью", "pr ", "deploy", "деплой", "сервер",
                          "пост", "telegram", "send message", "отправь",
                          "teams meeting"],
        "keywords": {
            "анализ": 3, "analysis": 3, "prism": 3, "аналитика": 3,
            "whatif": 3, "премортем": 3, "premortem": 3,
            "clarity": 3, "clarity-thinker": 3, "adversarial": 2,
            "security audit": 3, "threat model": 3,
            "codebase": 2, "audit": 2, "review": 2,
            "debt": 2, "technical debt": 3, "over-engineering": 3,
            "report": 2, "отчёт": 2, "summary": 2,
            "отчет": 3, "диатаксис": 3, "diataxis": 3,
            "классификация": 3, "strategy": 2, "стратегия": 2,
            "метрики": 2, "метрика": 2, "сравнение": 2,
            "подготовь отчет": 3, "подготовь отчёт": 3, "отчёт по проекту": 3,
            "отчет по проекту": 3, "проаналижи лог": 3, "лог ошибок": 3,
            "лог ошибок сервера": 5, "проаналижи лог ошибок": 5,
            "лог ошибок": 3, "проаналижи лог": 3,
            "сравни": 3, "vs": 3, "zvec": 2, "chroma": 2,
            "ошибка": 2, "ошибок": 2, "баг": 2, "лог": 2,
        },
        "collections": {
            "prism": ["prism", "prism-3way", "prism-discover", "prism-full"],
            "whatif": ["whatif", "premortem", "премортем", "риск", "risk"],
            "clarity": ["clarity-thinker", "clarity", "adversarial",
                       "security catalog"],
            "codebase": ["codebase", "audit", "technical debt", "debt",
                        "over-engineering", "ponytail"],
            "analysis": ["анализ", "analysis", "сравнение", "summary",
                        "polymarket", "market research",
                        "сравни", "подготовь отчет", "подготовь отчёт",
                        "проанализируй лог", "лог ошибок", "отчет по проекту",
                        "сравни zvec", "сравни chroma", "zvec vs chroma"],
        },
    },
    "creative": {
        "weight": 2,
        "anti_keywords": ["код", "ревью", "bug", "ошибка", "деплой", "сервер",
                          "тест", "test", "security", "firewall", "отправь",
                          "send message", "постинг", "api", "интеграция",
                          "github", "linear", "notion",
                          "adr", "архитектурное решение",
                          "google docs", "google forms", "google workspace",
                          "linear управления задачами"],
        "keywords": {
            "диаграмма": 3, "diagram": 3, "ascii": 3, "ascii art": 3,
            "comic": 3, "комикс": 3, "infographic": 3, "инфографика": 3,
            "видео": 3, "video": 3, "manim": 3, "animation": 3, "анимация": 3,
            "видео-анимация": 3, "видео анимация": 3, "создай видео": 3,
            "музыка": 3, "music": 3, "song": 3, "songwriting": 3, "suno": 3,
            "песня": 3, "песню": 3, "напиши песню": 3,
            "рисунок": 3, "drawing": 3, "sketch": 2, "pixel art": 3,
            "design": 2, "prototype": 3, "landing": 2,
            "excalidraw": 3, "p5.js": 3, "p5": 3,
            "illustration": 3, "illustrator": 3,
            "html": 1, "css": 1, "svg": 2,
            "ascii баннер": 3, "ascii banner": 3,
            "пиксель-арт": 3, "pixel art": 3,
            "нарисуй": 3, "создай рисунок": 3,
            "сгенерируй комикс": 3, "создай комикс": 3,
            "нарисуй пиксель-арт": 3,
        },
        "collections": {
            "diagrams": ["диаграмма", "diagram", "ascii", "ascii art",
                        "excalidraw", "schema diagram",
                        "ascii баннер", "ascii banner",
                        "ascii баннер для", "создай ascii баннер"],
            "comics_infographics": ["comic", "комикс", "infographic",
                                    "инфографика", "baoyu",
                                    "сгенерируй комикс", "создай комикс"],
            "video": ["видео", "video", "manim", "animation", "анимация",
                     "ascii video"],
            "music": ["музыка", "music", "song", "songwriting", "suno",
                     "audiocraft", "musicgen",
                     "песня", "песню", "напиши песню", "сгенерируй песню"],
            "visual": ["рисунок", "drawing", "sketch", "pixel art",
                       "illustration", "prototype", "design",
                       "пиксель-арт", "нарисуй", "нарисуй пиксель-арт",
                       "ascii art", "ascii баннер"],
        },
    },
    "personal": {
        "weight": 1,
        "keywords": {
            "ford": 2, "форд": 2, "explorer": 2, "ford explorer": 3,
            "диагностика ford": 3, "автодиагностика ford": 3,
            "здоровье": 3, "health": 3, "sibionics": 3,
            "заметки": 2, "notes": 2, "obsidian": 3, "note": 1,
            "голос": 2, "voice": 2, "tts": 2, "memo": 2,
            "озвучь": 3, "озвучу": 3, "озвучивание": 3,
            "sbl": 2, "system boundary": 2,
            "personal": 2, "личный": 2,
            "адаптация": 2, "adaptation": 2,
        },
        "collections": {
            "auto": ["ford", "форд", "explorer", "ford explorer",
                    "car", "машина", "automotive", "carpc",
                    "автодиагностика", "диагностика ford",
                    "найди информацию про ford", "запчасти для ford"],
            "health": ["здоровье", "health", "sibionics", "мониторинг здоровья",
                      "проверь уровень сахара", "сахар в sibionics"],
            "notes": ["заметки", "notes", "obsidian", "note taking", "voice memo",
                     "создай заметку", "озвучь", "tts", "текст через tts",
                     "озвучь текст"],
            "adaptation": ["адаптация", "adaptation", "yar", "ярославская"],
        },
    },
    "communication": {
        "weight": 2,
        "anti_keywords": ["статья", "постинг", "публикация", "оформи пост",
                          "напиши пост", "пост для", "напиши статью",
                          "контент", "style", "style_guide"],
        "keywords": {
            "сообщение": 3, "message": 3, "уведомление": 3,
            "telegram": 3, "телеграм": 3, "чат": 2, "chat": 2,
            "matrix": 3, "synapse": 3,
            "discord": 3, "slack": 2,
            "email": 3, "почта": 3, "imap": 2, "smtp": 2,
            "rich message": 3, "rich": 2,
            "отправь сообщение": 3, "send message": 3,
            "оповещение": 3, "notification": 3,
        },
        "collections": {
            "telegram": ["telegram", "телеграм", "rich message", "пост",
                        "posting", "chat"],
            "matrix": ["matrix", "synapse", "homeserver", "element"],
            "discord": ["discord", "bot", "server"],
            "email": ["email", "почта", "imap", "smtp", "himalaya",
                     "mail", "письмо"],
        },
    },
    "security": {
        "weight": 3,
        "anti_keywords": ["код-ревью", "refactor", "деплой кода",
                          "wiki страница", "пост в телеграм",
                          "тестирование", "test coverage", "unit test",
                          "astra linux", "astralinux", "альт", "альт линукс",
                          "мандатный доступ", "замкнутая среда", "патч"],
        "keywords": {
            "security": 3, "безопасность": 3, "аудит": 2, "audit": 2,
            "sbl": 3, "system boundary": 3, "boundary layer": 3,
            "sanitize": 3, "санитизация": 3, "forensics": 3,
            "уязвимость": 3, "vulnerability": 3, "cve": 3,
            "exploit": 3, "threat": 2, "threat model": 3,
            "injection": 3, "xss": 3, "sql injection": 3,
            "jailbreak": 3, "godmode": 3, "red team": 4, "red team тестирование": 5,
            "supply chain": 3, "зависимости": 2, "dependencies": 2,
            "проведи аудит": 3, "проанализируй безопасность": 3,
            "восстанови": 3, "восстановление": 3, "инцидент": 3,
        },
        "collections": {
            "audit": ["audit", "аудит", "sbl", "system boundary",
                     "boundary layer", "scan", "проведи аудит",
                     "проанализируй безопасность"],
            "forensics": ["forensics", "forensic", "evidence recovery",
                         "supply chain investigation",
                         "восстанови", "восстановление", "данные после",
                         "инцидент", "инцидента"],
            "hardening": ["sanitize", "санитизация", "vulnerability",
                         "уязвимость", "cve", "exploit", "hardening",
                         "firewall", "firewall правила", "настрой firewall",
                         "ufw", "ufw правила", "iptables", "iptables правила"],
            "red_team": ["jailbreak", "godmode", "red team", "threat",
                        "threat model", "injection", "red team тестирование"],
        },
    },
}

# ─── Нормализация ──────────────────────────────────────────────────

# ─── Стемминг (упрощённый) ─────────────────────────────────────────

# Русские окончания для нормализации
RUSSIAN_SUFFIXES = [
    'а', 'я', 'ы', 'и', 'е', 'у', 'ю', 'ой', 'ей', 'ом', 'ем', 'ах', 'ях',
    'ов', 'ев', 'ам', 'ям', 'ах', 'ях', 'ий', 'ый', 'ой', 'его', 'его',
    'ему', 'ему', 'им', 'ым', 'ем', 'ом', 'их', 'ых', 'ую', 'юю', 'ая', 'яя',
    'ое', 'ее', 'ые', 'ие', 'ый', 'ой', 'ого', 'ому', 'ими', 'ыми',
    'овать', 'евать', 'ать', 'ить', 'еть', 'ыть', 'уть', 'ять',
    'ция', 'циями', 'циях', 'циям',
    'ние', 'ния', 'ний', 'ниям', 'ниях',
    'ать', 'ает', 'ают', 'аем', 'аете', 'аешь',
    'ить', 'ит', 'ят', 'им', 'ите', 'ишь',
    'еть', 'еет', 'еют', 'еем', 'еете', 'ешь',
    'овать', 'ует', 'уют', 'уем', 'уете', 'уешь',
]

def _stem(word: str) -> str:
    """Упрощённый русский стеммер — возвращает основу слова."""
    if len(word) <= 3:
        return word
    for suffix in RUSSIAN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _stem_match(token: str, keywords: dict) -> tuple:
    """Проверяет совпадение токена с ключевыми словами (включая стемминг).
    
    Returns:
        (matched_keyword, weight) если совпадение, ("", 0) если нет.
    """
    # Прямое совпадение
    if token in keywords:
        return token, keywords[token]
    
    # Проверяем стемму
    stem = _stem(token)
    for kw, weight in keywords.items():
        if stem == _stem(kw):
            return kw, weight
    
    return "", 0


# Стоп-слова (не являются триггерами)
STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "this", "that", "these", "those", "it", "its", "and", "or", "but",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "please", "thanks", "thank", "hi", "hello", "hey",
}

# ─── Основные функции ──────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Нормализация текста: lowercase, удаление пунктуации."""
    text = text.lower()
    # Сохраняем дефисы внутри слов (code-review, tdd, etc.)
    text = re.sub(r'[^\w\s\-./]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_tokens(text: str) -> list[str]:
    """Извлечение токенов из текста с сохранением многословных фраз."""
    normalized = _normalize(text)
    # Сначала пробуем многословные фразы (2-4 слова)
    tokens = []
    words = normalized.split()
    
    i = 0
    while i < len(words):
        matched = False
        # Пробуем фразы длины 4, 3, 2
        for length in [4, 3, 2]:
            if i + length <= len(words):
                phrase = ' '.join(words[i:i + length])
                # Проверяем есть ли фраза в каком-либо домене
                for domain_data in DOMAIN_KEYWORDS.values():
                    if phrase in domain_data["keywords"]:
                        tokens.append(phrase)
                        i += length
                        matched = True
                        break
                if matched:
                    break
        if not matched:
            word = words[i]
            if word not in STOP_WORDS and len(word) > 1:
                tokens.append(word)
            i += 1
    
    return tokens


def classify(query: str) -> dict:
    """Классифицировать запрос по домену и коллекции.
    
    Args:
        query: Пользовательский запрос.
        
    Returns:
        {
            "domain": str,
            "collection": str,
            "confidence": float (0.0 - 1.0),
            "keywords_matched": list[str],
            "fallback": bool,
        }
    """
    tokens = _extract_tokens(query)
    if not tokens:
        return {
            "domain": "software-dev",
            "collection": "planning",
            "confidence": 0.0,
            "keywords_matched": [],
            "fallback": True,
        }
    
    # 1. Подсчёт весов для каждого домена (с учётом стемминга и анти-триггеров)
    domain_scores = {}
    domain_matched_keywords = {}
    
    # Нормализованный текст запроса для проверки анти-триггеров
    normalized_query_lower = _normalize(query)
    
    for domain_name, domain_data in DOMAIN_KEYWORDS.items():
        score = 0
        matched = []
        for token in tokens:
            kw, weight = _stem_match(token, domain_data["keywords"])
            if weight > 0:
                score += weight
                matched.append(kw)
        
        # Вычитаем очки за анти-триггеры
        anti_keywords = domain_data.get("anti_keywords", [])
        for anti_kw in anti_keywords:
            if anti_kw in normalized_query_lower:
                score -= 3
        
        if score > 0:
            domain_scores[domain_name] = score
            domain_matched_keywords[domain_name] = matched
    
    if not domain_scores:
        return {
            "domain": "software-dev",
            "collection": "planning",
            "confidence": 0.0,
            "keywords_matched": [],
            "fallback": True,
        }
    
    # 2. Выбор лучшего домена
    best_domain = max(domain_scores, key=lambda k: domain_scores[k])
    best_score = domain_scores[best_domain]
    matched_keywords = domain_matched_keywords[best_domain]
    
    # 3. Расчёт confidence
    # Нормализуем: если вес домена × количество совпавших токенов / общее количество токенов
    max_possible = len(tokens) * 3  # максимальный вес токена = 3
    confidence = min(best_score / max(max_possible, 1), 1.0)
    
    # 4. Определение коллекции внутри домена
    best_collection = None
    best_collection_score = 0
    normalized_query = _normalize(query)
    
    for collection_name, collection_keywords in DOMAIN_KEYWORDS[best_domain]["collections"].items():
        score = 0
        for token in tokens:
            if token in collection_keywords:
                score += 2
        # Проверяем многословные фразы из оригинального запроса
        for ck in collection_keywords:
            if ck in normalized_query:
                score += 3
        # Стемминг для коллекций
        for ck in collection_keywords:
            for kw in ck.split():
                stemmed_kw = _stem(kw)
                for token in tokens:
                    if stemmed_kw and len(stemmed_kw) > 2 and _stem(token) == stemmed_kw:
                        score += 1
        if score > best_collection_score:
            best_collection_score = score
            best_collection = collection_name
    
    # 5. Fallback decision
    # Снижаем порог: 0.2 вместо 0.35 (меньше fallback, выше recall)
    fallback = confidence < 0.2 or best_collection is None
    
    if fallback:
        return {
            "domain": best_domain,
            "collection": list(DOMAIN_KEYWORDS[best_domain]["collections"].keys())[0] if DOMAIN_KEYWORDS[best_domain]["collections"] else "general",
            "confidence": round(confidence, 2),
            "keywords_matched": matched_keywords,
            "fallback": True,
        }
    
    return {
        "domain": best_domain,
        "collection": best_collection or list(DOMAIN_KEYWORDS[best_domain]["collections"].keys())[0],
        "confidence": round(confidence, 2),
        "keywords_matched": matched_keywords,
        "fallback": False,
    }


def classify_with_details(query: str) -> dict:
    """Расширенная классификация с детальной диагностикой."""
    result = classify(query)
    tokens = _extract_tokens(query)
    
    # Добавляем scores всех доменов для отладки
    all_scores = {}
    for domain_name, domain_data in DOMAIN_KEYWORDS.items():
        score = 0
        for token in tokens:
            if token in domain_data["keywords"]:
                score += domain_data["keywords"][token]
        if score > 0:
            all_scores[domain_name] = score
    
    result["all_domain_scores"] = all_scores
    result["tokens"] = tokens
    return result


# ─── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ")
    
    result = classify_with_details(query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
