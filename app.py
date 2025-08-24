import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# --- Импорты LangChain ---
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.callbacks import get_openai_callback

# --- 1. Инициализация и настройка ---

# Загружаем переменные окружения
load_dotenv()

# Получаем настройки из переменных окружения
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")
MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b:free")
BRAND_NAME = os.getenv("BRAND_NAME", "Shoply")

# Проверки ключей и URL
if not API_KEY or not API_BASE:
    print("Ошибка: OPENAI_API_KEY и OPENAI_API_BASE должны быть установлены в .env файле.")
    exit()

# Создаём директорию для логов, если её нет
os.makedirs("logs", exist_ok=True)

# Настройка логгера для записи сессий в JSONL
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/session_{session_timestamp}.jsonl"
logger = logging.getLogger('chat_session')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# --- 2. Загрузка данных (FAQ и Заказы) ---

def load_data(filename):
    """Загружает JSON данные из файла."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка при загрузке файла {filename}: {e}")
        return None


faq_data = load_data("data/faq.json")
orders_data = load_data("data/orders.json")

if faq_data is None or orders_data is None:
    exit()


# --- 3. Функции для команд и FAQ ---

def get_order_status(order_id: str) -> str:
    """Возвращает статус заказа по ID или сообщение об ошибке."""
    order = orders_data.get(order_id)
    if not order:
        return f"Заказ с номером {order_id} не найден. Пожалуйста, проверьте номер."

    status = order.get("status")
    if status == "in_transit":
        return (f"Заказ #{order_id} в пути. "
                f"Примерный срок доставки: {order.get('eta_days')} дня. "
                f"Перевозчик: {order.get('carrier')}.")
    elif status == "delivered":
        return f"Заказ #{order_id} доставлен {order.get('delivered_at')}."
    elif status == "processing":
        return f"Заказ #{order_id} в обработке. {order.get('note')}."

    return f"Неизвестный статус для заказа #{order_id}."


def find_in_faq(question: str) -> str | None:
    """Ищет точный ответ в FAQ."""
    for item in faq_data:
        if question.lower().strip() == item["q"].lower().strip():
            return item["a"]
    return None


def log_entry(role: str, content: str, usage: dict = None):
    """Записывает событие в лог-файл."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content
    }
    if usage:
        entry["usage"] = usage
    logger.info(json.dumps(entry, ensure_ascii=False))


# --- 4. Основной цикл чат-бота с LangChain ---

def main():
    """Главная функция, запускающая цикл диалога."""

    # --- Инициализация модели и цепочки LangChain ---
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.2,
        request_timeout=30  # Увеличим таймаут для кастомных моделей
    )

    # Создаём цепочку с памятью
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        verbose=False  # Отключаем вывод дебаг-информации от LangChain в консоль
    )

    # Системный промпт для модели
    system_prompt = (
        f"Ты — ассистент поддержки магазина «{BRAND_NAME}». "
        "Отвечай кратко, вежливо и по делу. "
        "Не придумывай информацию, которой нет в предоставленном контексте. "
        "Используй FAQ для ответов на общие вопросы."
    )

    # Добавляем системное сообщение в память диалога
    conversation.memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    # Логируем системное сообщение
    log_entry("system", system_prompt)

    print(f"Чат-бот поддержки магазина «{BRAND_NAME}» запущен!")
    print("Используется модель:", MODEL_NAME)
    print("Введите ваш вопрос или команду /order <id>. Для выхода введите 'выход'.\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except KeyboardInterrupt:
            print("\n[Прерывание пользователем]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("выход", "quit", "exit"):
            print("Бот: До свидания!")
            log_entry("system", "User initiated exit. Session ended.")
            break

        log_entry("user", user_input)

        bot_reply = ""
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 1. Сначала обрабатываем команду /order
        if user_input.lower().startswith("/order"):
            parts = user_input.split()
            bot_reply = get_order_status(parts[1]) if len(
                parts) == 2 else "Неверный формат команды. Используйте: /order <номер заказа>"

        # 2. Затем ищем точное совпадение в FAQ
        else:
            faq_answer = find_in_faq(user_input)
            if faq_answer:
                bot_reply = faq_answer

            # 3. Если ничего не найдено, обращаемся к LLM через LangChain
            else:
                try:
                    # Собираем контекст из FAQ для передачи в модель
                    faq_context = "\n".join([f"Вопрос: {item['q']}\nОтвет: {item['a']}" for item in faq_data])
                    # Формируем входные данные для модели, добавляя контекст
                    input_for_llm = (
                        f"Используя следующий контекст из FAQ, ответь на вопрос пользователя.\n"
                        f"--- Контекст FAQ ---\n{faq_context}\n--- Конец контекста ---\n\n"
                        f"Вопрос пользователя: {user_input}"
                    )

                    # Используем get_openai_callback для подсчета токенов
                    with get_openai_callback() as cb:
                        bot_reply = conversation.predict(input=input_for_llm)
                        usage_data = {
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_tokens": cb.total_tokens
                        }

                except Exception as e:
                    bot_reply = f"[Ошибка] Не удалось получить ответ от модели: {e}"

        # Вывод и логирование ответа
        bot_reply = bot_reply.strip()
        print(f"Бот: {bot_reply}")
        log_entry("assistant", bot_reply, usage_data)


if __name__ == "__main__":
    main()