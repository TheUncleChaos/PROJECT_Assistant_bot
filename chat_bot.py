import warnings
import os

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠ Установи requests и beautifulsoup4 для поиска в интернете")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠ Установи duckduckgo-search для улучшенного поиска")

try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False
    print("⚠ Установи wikipedia-api для поиска в Wikipedia")


class WebSearch:
    """Класс для задачи функционала для поиска информации в интернете"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "web_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "search_cache.json"
        self.cache = self._load_cache()

        if WIKI_AVAILABLE:
            self.wiki = wikipediaapi.Wikipedia(
                user_agent='StrictAssistantBot/1.0',
                language='ru'
            )
        else:
            self.wiki = None

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: dict, max_age_hours: int = 24) -> bool:
        if "timestamp" not in cache_entry:
            return False
        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        age = (datetime.now() - cached_time).total_seconds() / 3600
        return age < max_age_hours

    def search_duckduckgo(self, query: str, max_results: int = 3) -> list[dict]:
        """Поиск в DuckDuckGo"""
        if not DDGS_AVAILABLE:
            return []

        cache_key = f"ddg_{self._get_cache_key(query)}"

        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            return self.cache[cache_key]["results"]

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, region='ru-ru', max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", "")
                    })

            self.cache[cache_key] = {
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            self._save_cache()

            return results
        except Exception as e:
            print(f"Ошибка DuckDuckGo: {e}")
            return []

    def search_wikipedia(self, query: str) -> dict | None:
        """Поиск в Wikipedia"""
        if not WIKI_AVAILABLE or not self.wiki:
            return None

        cache_key = f"wiki_{self._get_cache_key(query)}"

        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key], max_age_hours=168):
            return self.cache[cache_key]["result"]

        try:
            page = self.wiki.page(query)

            if not page.exists():
                search_results = self.wiki.search(query, results=3) if hasattr(self.wiki, 'search') else []
                for title in search_results:
                    page = self.wiki.page(title)
                    if page.exists():
                        break
                else:
                    return None

            summary = page.summary[:1000] if len(page.summary) > 1000 else page.summary

            result = {
                "title": page.title,
                "summary": summary,
                "url": page.fullurl
            }

            self.cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            self._save_cache()

            return result
        except Exception as e:
            print(f"Ошибка Wikipedia: {e}")
            return None

    def fetch_webpage(self, url: str, timeout: int = 10) -> str | None:
        """Получение текста с страницы сайта"""
        if not REQUESTS_AVAILABLE:
            return None

        cache_key = f"page_{self._get_cache_key(url)}"

        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key], max_age_hours=12):
            return self.cache[cache_key]["content"]

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)[:3000]

            self.cache[cache_key] = {
                "content": text,
                "timestamp": datetime.now().isoformat()
            }
            self._save_cache()

            return text
        except Exception as e:
            print(f"Ошибка загрузки страницы: {e}")
            return None

    def search(self, query: str, sources: list[str] = None) -> dict:
        """Поиск по различным источникам"""
        if sources is None:
            sources = ["wikipedia", "duckduckgo"]

        results = {
            "query": query,
            "wikipedia": None,
            "web_results": [],
            "found": False
        }

        if "wikipedia" in sources:
            wiki_result = self.search_wikipedia(query)
            if wiki_result:
                results["wikipedia"] = wiki_result
                results["found"] = True

        if "duckduckgo" in sources:
            ddg_results = self.search_duckduckgo(query)
            if ddg_results:
                results["web_results"] = ddg_results
                results["found"] = True

        return results

    def clear_cache(self):
        """Очистка кэша"""
        self.cache = {}
        self._save_cache()
        print("✓ Кэш поиска очищен")


class DynamicFunctions:
    """Класс задающий контейнер для добавляемых функций"""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self._functions: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}
        self._code: dict[str, str] = {}
        self._triggers: dict[str, list[str]] = {}

    def register(self, name: str, func: Callable, description: str, code: str, triggers: list[str] = None):
        self._functions[name] = func
        self._descriptions[name] = description
        self._code[name] = code
        self._triggers[name] = triggers or []

    def call(self, name: str, *args, **kwargs) -> Any:
        if name in self._functions:
            return self._functions[name](self.bot, *args, **kwargs)
        raise ValueError(f"Функция '{name}' не найдена")

    def exists(self, name: str) -> bool:
        return name in self._functions

    def list_all(self) -> dict[str, str]:
        return self._descriptions.copy()

    def get_code(self, name: str) -> str:
        return self._code.get(name, "")

    def get_triggers(self, name: str) -> list[str]:
        return self._triggers.get(name, [])

    def find_by_trigger(self, message: str) -> str | None:
        msg_lower = message.lower()
        for name, triggers in self._triggers.items():
            for trigger in triggers:
                if trigger.lower() in msg_lower:
                    return name
        return None

    def remove(self, name: str) -> bool:
        if name in self._functions:
            del self._functions[name]
            del self._descriptions[name]
            del self._code[name]
            del self._triggers[name]
            return True
        return False


class StrictAssistantBot:
    """
    То чем должен быть бот и его стиль общения:
    Роль: Ассистент с функцией поиска в интернете
    Элементы функционала/"личности":
    1) Не придумывает информацию
    2) Ищет ответы в интернете
    3) Самомодифицируется (если точнне, должен самомодифицироваться, по факту же дополняется пользователем прямо во время диалога)
    4) Кэширует результаты
    """

    VERSION = "0.01" #Так как, из-за того, что пришлось переписывать весь код, по причинам удаления первоначального, эта версия не дотягивает лаже до "1.0"

    def __init__(
            self,
            model_name: str = "ai-forever/rugpt3medium_based_on_gpt2",
            data_dir: str = "assistant_data"
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.config_path = self.data_dir / "config.json"
        self.dictionary_path = self.data_dir / "dictionary.json"
        self.qa_pairs_path = self.data_dir / "qa_pairs.json"
        self.style_path = self.data_dir / "style.json"
        self.functions_path = self.data_dir / "functions.json"
        self.knowledge_path = self.data_dir / "knowledge.json"
        self.history_path = self.data_dir / "conversation_history.json"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Устройство: {self.device.upper()}")

        self.model_name = model_name

        self.config = self._load_json(self.config_path, {
            "user_name": "",
            "strict_mode": True,
            "web_search_enabled": True,
            "auto_search": True,
            "max_response_length": 200,
            "search_cache_hours": 24,
            "save_history": True
        })

        self.dictionary = self._load_json(self.dictionary_path, {})
        self.qa_pairs = self._load_json(self.qa_pairs_path, [])
        self.style = self._load_json(self.style_path, {
            "tone": "нейтральный, информативный и точный",
            "examples": [],
            "rules": [
                "Отвечай только на заданный вопрос",
                "Не придумывай факты — ищи информацию",
                "Если не знаешь — скажи об этом или поищи в интернете",
                "Указывай источники информации",
                "Уточняй если вопрос неясен"
            ]
        })
        self.knowledge = self._load_json(self.knowledge_path, {})

        self.web_search = WebSearch(self.data_dir)

        self.dynamic = DynamicFunctions(self)
        self._load_custom_functions()
        self._register_builtin_functions()

        self.conversation_history = []
        if self.config["save_history"]:
            saved_history = self._load_json(self.history_path, [])
            self.conversation_history = saved_history[-50:]

        self.awaiting_code_input = False
        self.pending_function = {"name": "", "desc": "", "triggers": []}
        self.last_search_results = None

        self._load_model()

    def _load_json(self, path: Path, default) -> dict | list:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return default if isinstance(default, (dict, list)) else {}

    def _save_json(self, path: Path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_config(self):
        self._save_json(self.config_path, self.config)

    def _save_history(self):
        if self.config["save_history"]:
            self._save_json(self.history_path, self.conversation_history[-100:])

    def _load_model(self):
        print(f"\nЗагрузка: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✓ Модель готова к работе\n")

    def _load_custom_functions(self):
        saved = self._load_json(self.functions_path, {})

        for name, data in saved.items():
            try:
                exec_globals = self._get_exec_globals()
                exec(data["code"], exec_globals)

                func_name = re.search(r'def\s+(\w+)\s*\(', data["code"])
                if func_name:
                    actual_name = func_name.group(1)
                    if actual_name in exec_globals:
                        self.dynamic.register(
                            name,
                            exec_globals[actual_name],
                            data.get("description", ""),
                            data["code"],
                            data.get("triggers", [])
                        )
            except Exception as e:
                print(f"⚠ Ошибка загрузки функции {name}: {e}")

    def _save_custom_functions(self):
        data = {}
        for name in self.dynamic._functions:
            if name not in self._builtin_function_names:
                data[name] = {
                    "description": self.dynamic._descriptions.get(name, ""),
                    "code": self.dynamic._code.get(name, ""),
                    "triggers": self.dynamic._triggers.get(name, [])
                }
        self._save_json(self.functions_path, data)

    def _get_exec_globals(self) -> dict:
        return {
            "bot": self,
            "datetime": datetime,
            "json": json,
            "os": os,
            "Path": Path,
            "re": re,
            "requests": requests if REQUESTS_AVAILABLE else None,
            "BeautifulSoup": BeautifulSoup if REQUESTS_AVAILABLE else None
        }

    def _register_builtin_functions(self):
        """Задача встроенных функций"""
        self._builtin_function_names = set()

        # Изначально данная функция была тестовой и задавалась самим ботом, но, в итоге было принято решение поместить её в встроенные
        def get_time(bot, *args):
            now = datetime.now()
            return f"Сейчас {now.strftime('%H:%M:%S')}, {now.strftime('%d.%m.%Y')}"

        self.dynamic.register(
            "get_time", get_time,
            "Показать текущее время и дату",
            "# встроенная функция",
            ["время", "который час", "какое число", "дата", "сегодня"]
        )
        self._builtin_function_names.add("get_time")

        # Также, как и с функцией показа времени
        def calculate(bot, expression="", *args):
            if not expression:
                return "Укажи выражение для вычисления"
            try:
                allowed = set('0123456789+-*/.() ')
                expr = str(expression)
                if all(c in allowed for c in expr):
                    result = eval(expr)
                    return f"{expr} = {result}"
                return "Недопустимые символы"
            except Exception as e:
                return f"Ошибка: {e}"

        self.dynamic.register(
            "calculate", calculate,
            "Калькулятор",
            "# встроенная функция",
            ["посчитай", "вычисли", "калькулятор", "сколько будет"]
        )
        self._builtin_function_names.add("calculate")

        def web_search(bot, query="", *args):
            if not query:
                return "Укажи что искать"
            results = bot.web_search.search(query)
            return bot._format_search_results(results)

        self.dynamic.register(
            "web_search", web_search,
            "Поиск в интернете",
            "# встроенная функция",
            ["найди в интернете", "загугли", "поищи", "что такое"]
        )
        self._builtin_function_names.add("web_search")

        # Так как это ассистент, то он должен иметь возможность сохранять что-либо в виде заметок
        def manage_notes(bot, action="list", text="", *args):
            notes_file = bot.data_dir / "notes.json"
            notes = bot._load_json(notes_file, [])

            if action == "add" and text:
                notes.append({"text": text, "date": datetime.now().isoformat()})
                bot._save_json(notes_file, notes)
                return f"✓ Заметка сохранена: {text}"
            elif action == "list":
                if not notes:
                    return "Заметок нет"
                result = "📝 Заметки:\n"
                for i, n in enumerate(notes[-10:], 1):
                    result += f"{i}. {n['text']}\n"
                return result
            elif action == "clear":
                bot._save_json(notes_file, [])
                return "✓ Заметки очищены"
            return "Действия: add, list, clear"

        self.dynamic.register(
            "manage_notes", manage_notes,
            "Управление заметками",
            "# встроенная функция",
            ["заметка", "запиши", "запомни", "заметки"]
        )
        self._builtin_function_names.add("manage_notes")

    def _should_search_web(self, message: str, intent: dict) -> bool:
        """Определяет, нужен ли поиск в интернете"""
        if not self.config["web_search_enabled"]:
            return False

        if not self.config["auto_search"]:
            return False

        search_triggers = [
            "найди", "поищи", "загугли", "что такое", "кто такой",
            "расскажи про", "информация о", "узнай", "в интернете"
        ]

        msg_lower = message.lower()
        for trigger in search_triggers:
            if trigger in msg_lower:
                return True

        fact_patterns = [
            r"когда (был|была|было|родился|умер|создан)",
            r"где (находится|расположен)",
            r"сколько (стоит|весит|длится)",
            r"почему (так|это)",
            r"как (работает|устроен|сделать)"
        ]

        for pattern in fact_patterns:
            if re.search(pattern, msg_lower):
                return True

        return False

    def _extract_search_query(self, message: str) -> str:
        """Функция дающая возможность боту извлекать запрос на поиск из сообщения"""
        patterns = [
            r"(?:найди|поищи|загугли|узнай)\s+(?:информацию\s+)?(?:о|про|об)?\s*(.+)",
            r"что такое\s+(.+)",
            r"кто такой\s+(.+)",
            r"расскажи\s+(?:мне\s+)?(?:о|про|об)\s+(.+)",
            r"информация\s+(?:о|про|об)\s+(.+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).strip().rstrip('?.,!')

        stop_words = ['пожалуйста', 'можешь', 'скажи', 'мне', 'ли', 'это', 'а', 'и', 'в', 'на']
        words = message.lower().split()
        query_words = [w for w in words if w not in stop_words and len(w) > 2]

        return ' '.join(query_words[:5])

    def _format_search_results(self, results: dict) -> str:
        """Функция форматирования результатов поиска"""
        if not results["found"]:
            return "К сожалению, ничего не найдено."

        parts = []

        if results["wikipedia"]:
            wiki = results["wikipedia"]
            parts.append(f"📚 **{wiki['title']}** (Wikipedia)")

            summary = wiki["summary"]
            if len(summary) > 500:
                summary = summary[:500] + "..."
            parts.append(summary)
            parts.append(f"🔗 {wiki['url']}\n")

        if results["web_results"]:
            if not results["wikipedia"]:
                parts.append("🌐 Результаты поиска:\n")

            for i, r in enumerate(results["web_results"][:3], 1):
                parts.append(f"{i}. **{r['title']}**")
                if r['snippet']:
                    snippet = r['snippet'][:200] + "..." if len(r['snippet']) > 200 else r['snippet']
                    parts.append(f"   {snippet}")
                if r['url']:
                    parts.append(f"   🔗 {r['url']}")
                parts.append("")

        return "\n".join(parts)

    def search_and_answer(self, query: str) -> str:
        """Поиск и формирование ответа"""
        print(f"🔍 Ищу: {query}")

        results = self.web_search.search(query)
        self.last_search_results = results

        if not results["found"]:
            return f"Я поискал информацию о '{query}', но ничего не нашёл. Попробуйте переформулировать запрос."

        return self._format_search_results(results)

    def _analyze_intent(self, message: str) -> dict:
        msg_lower = message.lower().strip()

        intent = {
            "type": "question",
            "needs_clarification": False,
            "is_creative": False,
            "is_function_request": False,
            "is_search_request": False,
            "confidence": 0.5
        }

        func_keywords = [
            "создай функцию", "добавь функцию", "научись",
            "добавь возможность", "создай команду"
        ]
        for kw in func_keywords:
            if kw in msg_lower:
                intent["type"] = "function_request"
                intent["is_function_request"] = True
                return intent

        creative_keywords = ["придумай", "сочини", "напиши историю", "пофантазируй"]
        for kw in creative_keywords:
            if kw in msg_lower:
                intent["type"] = "creative"
                intent["is_creative"] = True
                return intent

        if self._should_search_web(message, intent):
            intent["type"] = "search"
            intent["is_search_request"] = True
            return intent

        if "?" in message or any(w in msg_lower for w in ["как", "что", "где", "когда", "почему", "кто"]):
            intent["type"] = "question"
            intent["confidence"] = 0.8

        if len(message.split()) < 2:
            intent["needs_clarification"] = True

        return intent

    def _find_in_knowledge(self, message: str) -> str | None:
        msg_lower = message.lower().strip()

        for qa in self.qa_pairs:
            if qa["question"].lower().strip() == msg_lower:
                return qa["answer"]

        best_match = None
        best_score = 0

        for qa in self.qa_pairs:
            q_words = set(qa["question"].lower().split())
            m_words = set(msg_lower.split())

            if q_words:
                score = len(q_words & m_words) / len(q_words)
                if score > best_score and score >= 0.7:
                    best_score = score
                    best_match = qa["answer"]

        if best_match:
            return best_match

        for topic, info in self.knowledge.items():
            if topic.lower() in msg_lower:
                return f"📖 {topic}: {info}"

        for term, definition in self.dictionary.items():
            if term.lower() in msg_lower and msg_lower.startswith(("что такое", "что значит")):
                return f"📖 {term}: {definition}"

        return None


    def _handle_function_request(self, message: str) -> str:
        patterns = [
            r"(?:создай|добавь)\s+(?:функцию|команду)[:\s]*(.+)",
            r"научись\s+(.+)",
            r"добавь возможность\s+(.+)"
        ]

        description = ""
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                description = match.group(1).strip()
                break

        if not description:
            description = message

        func_name = "custom_" + re.sub(r'[^a-z0-9]', '_', description.lower())[:25]
        func_name = re.sub(r'_+', '_', func_name).strip('_')

        code = self._generate_function_code(description)

        if code:
            return self._create_function(func_name, description, code, [])

        # Чуть выше прописана функция дающая боту возможность попытаться сгенерировать функцию по запросу пользователя, но она работает через раз, поэтому в случае неудачи бот будет запрашивать код функции у пользователя
        self.awaiting_code_input = True
        self.pending_function = {
            "name": func_name,
            "desc": description,
            "triggers": []
        }

        return (
            f"Понял, нужна функция: **{description}**\n\n"
            f"Я пока не знаю, как её реализовать.\n"
            f"Введи код на Python:\n\n"
            f"```python\n"
            f"def {func_name}(bot, *args):\n"
            f"    # твой код\n"
            f"    return 'результат'\n"
            f"```\n\n"
            f"Или /cancel для отмены."
        )

    def _generate_function_code(self, description: str) -> str | None:
        desc = description.lower()

        templates = {
            "время": '''
def get_current_time(bot, *args):
    from datetime import datetime
    now = datetime.now()
    return f"Сейчас {now.strftime('%H:%M:%S')}, {now.strftime('%d.%m.%Y')}"
''',
            "случайное значение": '''
def random_number(bot, min_val=1, max_val=100, *args):
    import random
    return f"Случайное число: {random.randint(int(min_val), int(max_val))}"
''',
            "перевернуть": '''
def reverse_text(bot, text="", *args):
    if not text:
        return "Укажи текст"
    return f"Перевёрнутый текст: {text[::-1]}"
''',
            "подсчитать длину": '''
def text_length(bot, text="", *args):
    if not text:
        return "Укажи текст"
    return f"Длина текста: {len(text)} символов, {len(text.split())} слов"
''',
            "в": '''
def to_upper(bot, text="", *args):
    if not text:
        return "Укажи текст"
    return text.upper()
''',
            "н": '''
def to_lower(bot, text="", *args):
    if not text:
        return "Укажи текст"
    return text.lower()
'''
        }

        for keyword, code in templates.items():
            if keyword in desc:
                return code

        return None

    def _create_function(self, name: str, description: str, code: str, triggers: list) -> str:
        try:
            match = re.search(r'def\s+(\w+)\s*\(', code)
            if not match:
                return "Ошибка: не найдено определение функции"

            actual_name = match.group(1)

            exec_globals = self._get_exec_globals()
            exec(code, exec_globals)

            if actual_name not in exec_globals:
                return f"Ошибка: функция {actual_name} не создана"

            self.dynamic.register(
                actual_name,
                exec_globals[actual_name],
                description,
                code,
                triggers
            )

            self._save_custom_functions()

            return f"✓ Функция **{actual_name}** создана!\nОписание: {description}"

        except SyntaxError as e:
            return f"Синтаксическая ошибка:\n{e}"
        except Exception as e:
            return f"Ошибка:\n{e}"

    def _process_code_input(self, code: str) -> str:
        self.awaiting_code_input = False

        if code.strip().lower() == "/cancel":
            self.pending_function = {"name": "", "desc": "", "triggers": []}
            return "Отменено."

        code = code.replace("```python", "").replace("```", "").strip()

        result = self._create_function(
            self.pending_function["name"],
            self.pending_function["desc"],
            code,
            self.pending_function["triggers"]
        )

        self.pending_function = {"name": "", "desc": "", "triggers": []}
        return result

    def _try_execute_function(self, message: str) -> str | None:
        msg_lower = message.lower()

        func_name = self.dynamic.find_by_trigger(message)

        if func_name:
            try:
                args = self._extract_function_args(message, func_name)
                result = self.dynamic.call(func_name, *args)
                return str(result)
            except Exception as e:
                return f"Ошибка функции: {e}"

        return None

    def _extract_function_args(self, message: str, func_name: str) -> list:
        msg = message.lower()
        args = []

        # Калькулятор
        if func_name == "calculate":
            expr = re.search(r'[\d\s\+\-\*\/\.\(\)]+', message)
            if expr:
                args = [expr.group().strip()]

        elif func_name == "manage_notes":
            if any(w in msg for w in ["добавь", "запиши", "запомни", "сохрани"]):
                text = re.sub(r'(добавь|запиши|запомни|сохрани|заметку?|в заметки)\s*', '', message, flags=re.I).strip()
                args = ["add", text]
            elif any(w in msg for w in ["покажи", "список", "все заметки"]):
                args = ["list"]
            elif any(w in msg for w in ["очисти", "удали"]):
                args = ["clear"]
            else:
                args = ["list"]

        elif func_name == "web_search":
            query = self._extract_search_query(message)
            args = [query]

        return args


    def _build_prompt(self, message: str, intent: dict, context: str = "") -> str:
        parts = []

        parts.append("=== ИНСТРУКЦИИ ===")
        parts.append("Ты — точный и полезный ассистент.")

        for rule in self.style["rules"]:
            parts.append(f"• {rule}")

        parts.append(f"• Тон: {self.style['tone']}")
        parts.append("• НЕ придумывай факты")
        parts.append("• НЕ говори от лица пользователя")

        if intent["is_creative"]:
            parts.append("• Пользователь просит творческий контент — можно фантазировать")

        if context:
            parts.append(f"\n=== НАЙДЕННАЯ ИНФОРМАЦИЯ ===\n{context[:800]}")

        if self.style["examples"]:
            parts.append("\n=== ПРИМЕРЫ ОТВЕТОВ ===")
            for ex in self.style["examples"][-2:]:
                parts.append(f"• {ex}")

        relevant = []
        for term, definition in self.dictionary.items():
            if term.lower() in message.lower():
                relevant.append(f"{term}: {definition}")
        if relevant:
            parts.append("\n=== ТЕРМИНЫ ===")
            for t in relevant[:3]:
                parts.append(f"• {t}")

        if self.config["user_name"]:
            parts.append(f"\nПользователь: {self.config['user_name']}")

        if self.conversation_history:
            parts.append("\n=== ИСТОРИЯ ===")
            for msg in self.conversation_history[-2:]:
                parts.append(f"Пользователь: {msg['user']}")
                parts.append(f"Ассистент: {msg['bot'][:100]}")

        parts.append(f"\n=== ЗАПРОС ===")
        parts.append(f"Пользователь: {message}")
        parts.append("Ассистент:")

        return "\n".join(parts)

    def _generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        max_input = 450
        if inputs.shape[1] > max_input:
            inputs = inputs[:, -max_input:]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config["max_response_length"],
                temperature=0.6,
                top_p=0.85,
                top_k=40,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.4,
                no_repeat_ngram_size=3
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Ассистент:" in full_output:
            response = full_output.split("Ассистент:")[-1].strip()
        else:
            response = full_output[len(prompt):].strip()

        # Очистка
        for marker in ["Пользователь:", "===", "User:", "\n\n\n"]:
            if marker in response:
                response = response.split(marker)[0].strip()

        return response


    def chat(self, message: str) -> str:
        if self.awaiting_code_input:
            return self._process_code_input(message)

        intent = self._analyze_intent(message)

        if intent["is_function_request"]:
            return self._handle_function_request(message)

        func_result = self._try_execute_function(message)
        if func_result:
            self._add_to_history(message, func_result)
            return func_result

        knowledge = self._find_in_knowledge(message)
        if knowledge:
            self._add_to_history(message, knowledge)
            return knowledge

        search_context = ""
        if intent["is_search_request"] and self.config["web_search_enabled"]:
            query = self._extract_search_query(message)
            if query:
                search_result = self.search_and_answer(query)
                self._add_to_history(message, search_result)
                return search_result

        if intent["needs_clarification"]:
            response = "Можешь уточнить вопрос? Хочу дать точный ответ."
            self._add_to_history(message, response)
            return response

        prompt = self._build_prompt(message, intent, search_context)
        response = self._generate_response(prompt)

        if len(response) < 3:
            response = "Не уверен, как ответить. Попробуй переформулировать или попроси поискать в интернете."

        if self.config["strict_mode"] and not intent["is_creative"]:
            uncertain = ["наверное", "возможно", "думаю", "кажется", "может быть"]
            if any(u in response.lower() for u in uncertain):
                response += "\n\n⚠️ Я не уверен в точности. Хочешь, поищу в интернете?"

        self._add_to_history(message, response)
        return response

    def _add_to_history(self, user_msg: str, bot_msg: str):
        self.conversation_history.append({
            "user": user_msg,
            "bot": bot_msg,
            "timestamp": datetime.now().isoformat()
        })
        self._save_history()


    def add_to_dictionary(self, term: str, definition: str):
        self.dictionary[term] = definition
        self._save_json(self.dictionary_path, self.dictionary)
        print(f"✓ {term}: {definition}")

    def add_qa_pair(self, question: str, answer: str):
        self.qa_pairs.append({"question": question, "answer": answer})
        self._save_json(self.qa_pairs_path, self.qa_pairs)
        print("✓ Q&A добавлена")

    def add_knowledge(self, topic: str, info: str):
        self.knowledge[topic] = info
        self._save_json(self.knowledge_path, self.knowledge)
        print(f"✓ Знание: {topic}")

    def set_style_tone(self, tone: str):
        self.style["tone"] = tone
        self._save_json(self.style_path, self.style)
        print(f"✓ Тон: {tone}")

    def add_style_rule(self, rule: str):
        self.style["rules"].append(rule)
        self._save_json(self.style_path, self.style)
        print("✓ Правило добавлено")

    def add_style_example(self, example: str):
        self.style["examples"].append(example)
        self._save_json(self.style_path, self.style)
        print("✓ Пример добавлен")

    def load_qa_from_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            pattern = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)'
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

            for q, a in matches:
                self.qa_pairs.append({"question": q.strip(), "answer": a.strip()})

            self._save_json(self.qa_pairs_path, self.qa_pairs)
            print(f"✓ Загружено {len(matches)} пар")
        except Exception as e:
            print(f"Ошибка: {e}")

    def show_functions(self):
        funcs = self.dynamic.list_all()
        if not funcs:
            print("Нет функций")
            return

        print("\n" + "=" * 45)
        print("           ФУНКЦИИ")
        print("=" * 45)
        for name, desc in funcs.items():
            triggers = self.dynamic.get_triggers(name)
            print(f"• {name}: {desc}")
            if triggers:
                print(f"  Триггеры: {', '.join(triggers[:3])}")
        print("=" * 45 + "\n")

    def show_status(self):
        print("\n" + "=" * 50)
        print("              СТАТУС")
        print("=" * 50)
        print(f"Версия: {self.VERSION}")
        print(f"Пользователь: {self.config['user_name'] or '—'}")
        print(f"Устройство: {self.device}")
        print(f"Строгий режим: {'✓' if self.config['strict_mode'] else '✗'}")
        print(f"Поиск в интернете: {'✓' if self.config['web_search_enabled'] else '✗'}")
        print(f"Автопоиск: {'✓' if self.config['auto_search'] else '✗'}")
        print(f"Тон: {self.style['tone']}")
        print(f"Терминов: {len(self.dictionary)}")
        print(f"Q&A пар: {len(self.qa_pairs)}")
        print(f"Знаний: {len(self.knowledge)}")
        print(f"Функций: {len(self.dynamic._functions)}")
        print(f"История: {len(self.conversation_history)} сообщений")
        print("=" * 50 + "\n")

    def reset_conversation(self):
        self.conversation_history = []
        self._save_history()
        print("✓ История очищена")

#Псевдо красивая иконка команд
def print_help():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                      ДОСТУПНЫЕ КОМАНДЫ                             ║
╠════════════════════════════════════════════════════════════════════╣
║  ОСНОВНЫЕ:                                                         ║
║    /help              — справка                                    ║
║    /status            — статус бота                                ║
║    /reset             — очистить историю                           ║
║    /quit              — выход                                      ║
╠════════════════════════════════════════════════════════════════════╣
║  НАСТРОЙКИ:                                                        ║
║    /set_name          — имя пользователя                           ║
║    /set_tone          — тон общения                                ║
║    /add_rule          — правило поведения                          ║
║    /add_style         — пример стиля                               ║
║    /strict_on/off     — строгий режим                              ║
║    /search_on/off     — поиск в интернете                          ║
║    /autosearch_on/off — автоматический поиск                       ║
╠════════════════════════════════════════════════════════════════════╣
║  ОБУЧЕНИЕ:                                                         ║
║    /dict_add          — термин в словарь                           ║
║    /dict_show         — показать словарь                           ║
║    /qa_add            — пара вопрос-ответ                          ║
║    /qa_load           — загрузить Q&A из файла                     ║
║    /qa_show           — показать Q&A                               ║
║    /knowledge_add     — добавить знание                            ║
║    /knowledge_show    — показать знания                            ║
╠════════════════════════════════════════════════════════════════════╣
║  ПОИСК:                                                            ║
║    /search <запрос>   — поиск в интернете                          ║
║    /wiki <запрос>     — поиск в Wikipedia                          ║
║    /clear_cache       — очистить кэш поиска                        ║
╠════════════════════════════════════════════════════════════════════╣
║  ФУНКЦИИ:                                                          ║
║    /func_list         — список функций                             ║
║    /func_code <имя>   — код функции                                ║
║    /func_add          — добавить функцию                           ║
║    /func_remove <имя> — удалить функцию                            ║
║                                                                    ║
║  Или в чате: "Создай функцию...", "Научись..."                     ║
╠════════════════════════════════════════════════════════════════════╣
║  ВСТРОЕННЫЕ ВОЗМОЖНОСТИ:                                           ║
║    • "Который час?" — время                                        ║
║    • "Посчитай 2+2" — калькулятор                                  ║
║    • "Запиши заметку..." — заметки                                 ║
║    • "Найди информацию о..." — поиск                               ║
╚════════════════════════════════════════════════════════════════════╝
""")


def main():
    print("\n" + "=" * 60)
    print("        СТРОГИЙ АССИСТЕНТ С ПОИСКОМ В ИНТЕРНЕТЕ")
    print("=" * 60)

    try:
        bot = StrictAssistantBot()
    except Exception as e:
        print(f"Ошибка: {e}")
        traceback.print_exc()
        return

    if not bot.config["user_name"]:
        name = input("Как тебя зовут? (Enter — пропустить): ").strip()
        if name:
            bot.config["user_name"] = name
            bot._save_config()
    else:
        print(f"Привет, {bot.config['user_name']}!")

    print("\n/help — список команд")
    print("Я могу искать информацию в интернете!\n")

    while True:
        try:
            user_input = input("Ты: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("До свидания!")
                break

            elif user_input == "/help":
                print_help()

            elif user_input == "/status":
                bot.show_status()

            elif user_input == "/reset":
                bot.reset_conversation()

            elif user_input == "/set_name":
                name = input("Имя: ").strip()
                if name:
                    bot.config["user_name"] = name
                    bot._save_config()
                    print(f"✓ Имя: {name}")

            elif user_input == "/set_tone":
                tone = input("Тон: ").strip()
                if tone:
                    bot.set_style_tone(tone)

            elif user_input == "/add_rule":
                rule = input("Правило: ").strip()
                if rule:
                    bot.add_style_rule(rule)

            elif user_input == "/add_style":
                ex = input("Пример: ").strip()
                if ex:
                    bot.add_style_example(ex)

            elif user_input == "/strict_on":
                bot.config["strict_mode"] = True
                bot._save_config()
                print("✓ Строгий режим ВКЛ")

            elif user_input == "/strict_off":
                bot.config["strict_mode"] = False
                bot._save_config()
                print("✓ Строгий режим ВЫКЛ")

            elif user_input == "/search_on":
                bot.config["web_search_enabled"] = True
                bot._save_config()
                print("✓ Поиск ВКЛ")

            elif user_input == "/search_off":
                bot.config["web_search_enabled"] = False
                bot._save_config()
                print("✓ Поиск ВЫКЛ")

            elif user_input == "/autosearch_on":
                bot.config["auto_search"] = True
                bot._save_config()
                print("✓ Автопоиск ВКЛ")

            elif user_input == "/autosearch_off":
                bot.config["auto_search"] = False
                bot._save_config()
                print("✓ Автопоиск ВЫКЛ")

            elif user_input == "/dict_add":
                term = input("Термин: ").strip()
                definition = input("Определение: ").strip()
                if term and definition:
                    bot.add_to_dictionary(term, definition)

            elif user_input == "/dict_show":
                if not bot.dictionary:
                    print("Словарь пуст")
                else:
                    print("\n=== СЛОВАРЬ ===")
                    for t, d in bot.dictionary.items():
                        print(f"• {t}: {d}")
                    print()

            elif user_input == "/qa_add":
                q = input("Вопрос: ").strip()
                a = input("Ответ: ").strip()
                if q and a:
                    bot.add_qa_pair(q, a)

            elif user_input == "/qa_load":
                path = input("Путь: ").strip()
                bot.load_qa_from_file(path)

            elif user_input == "/qa_show":
                if not bot.qa_pairs:
                    print("Нет Q&A")
                else:
                    print("\n=== Q&A ===")
                    for i, qa in enumerate(bot.qa_pairs[-10:], 1):
                        print(f"{i}. Q: {qa['question'][:40]}...")
                        print(f"   A: {qa['answer'][:40]}...\n")

            elif user_input == "/knowledge_add":
                topic = input("Тема: ").strip()
                info = input("Информация: ").strip()
                if topic and info:
                    bot.add_knowledge(topic, info)

            elif user_input == "/knowledge_show":
                if not bot.knowledge:
                    print("База знаний пуста")
                else:
                    print("\n=== ЗНАНИЯ ===")
                    for topic, info in bot.knowledge.items():
                        print(f"• {topic}: {info[:50]}...")
                    print()

            elif user_input.startswith("/search "):
                query = user_input[8:].strip()
                if query:
                    result = bot.search_and_answer(query)
                    print(f"Бот: {result}\n")

            elif user_input.startswith("/wiki "):
                query = user_input[6:].strip()
                if query:
                    result = bot.web_search.search_wikipedia(query)
                    if result:
                        print(f"\n📚 {result['title']}\n{result['summary'][:500]}...\n🔗 {result['url']}\n")
                    else:
                        print("Не найдено в Wikipedia\n")

            elif user_input == "/clear_cache":
                bot.web_search.clear_cache()

            elif user_input == "/func_list":
                bot.show_functions()

            elif user_input.startswith("/func_code "):
                name = user_input[11:].strip()
                code = bot.dynamic.get_code(name)
                if code and code != "# встроенная функция":
                    print(f"\n--- {name} ---\n{code}\n---\n")
                elif code:
                    print(f"{name} — встроенная функция\n")
                else:
                    print(f"Функция '{name}' не найдена\n")

            elif user_input == "/func_add":
                print("Введи код (пустая строка — конец):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)

                if lines:
                    code = "\n".join(lines)
                    desc = input("Описание: ").strip()
                    triggers = input("Триггеры через запятую: ").strip()
                    trigger_list = [t.strip() for t in triggers.split(",") if t.strip()]

                    bot.pending_function = {"name": "", "desc": desc, "triggers": trigger_list}
                    result = bot._process_code_input(code)
                    print(result)

            elif user_input.startswith("/func_remove "):
                name = user_input[13:].strip()
                if bot.dynamic.remove(name):
                    bot._save_custom_functions()
                    print(f"✓ Функция '{name}' удалена")
                else:
                    print(f"Функция '{name}' не найдена")

            else:
                response = bot.chat(user_input)
                print(f"Бот: {response}\n")

        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}\n")
            traceback.print_exc()


if __name__ == "__main__":
    main()