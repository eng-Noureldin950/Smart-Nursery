import telebot
import threading
from config import API_TOKEN

bot = telebot.TeleBot(API_TOKEN)

users = set()


@bot.message_handler(commands=["start"])
def welcome(message):
    chat_id = message.chat.id
    users.add(chat_id)

    print(f"User registered: {chat_id}")

    bot.send_message(
        chat_id,
        f"Hello {message.from_user.first_name}! 👋\n"
        "Welcome to Smart Nursery Guardian."
    )


def send_gas_alert():
    text = (
        "🚨 URGENT SAFETY ALERT!\n"
        "Smoke or gas has been detected in the nursery."
    )

    for chat_id in users:
        try:
            bot.send_message(chat_id, text)
        except Exception as e:
            print("Telegram send failed:", e)


def send_hungry_alert():
    for chat_id in users:
        try:
            bot.send_message(
                chat_id,
                "👶 Baby Alert\n"
                "The baby has been detected as HUNGRY."
            )
        except Exception as e:
            print("Telegram send failed:", e)


def send_tired_alert():
    for chat_id in users:
        try:
            bot.send_message(
                chat_id,
                "👶 Baby Alert\n"
                "The baby has been detected as TIRED.\n"
                "⚠️ Urgent care may be needed."
            )
        except Exception as e:
            print("Telegram send failed:", e)


def start_bot():
    bot.infinity_polling()


# Start Telegram polling in background
telegram_thread = threading.Thread(
    target=start_bot,
    daemon=True
)

telegram_thread.start()
