import telebot
from config import API_TOKEN

bot = telebot.TeleBot(API_TOKEN)

# Store users who started the bot
users = set()

#starting the bot

@bot.message_handler(commands=["start"])
def welcome(message):
    chat_id = message.chat.id
    # Add this user's chat ID
    users.add(message.chat.id)

    bot.send_message(
        message.chat.id,
        f"Hello {message.from_user.first_name}! 👋\n"
        "Welcome to Smart Nursery Guardian."
    )

def send_gas_alert():
    text = "🚨 URGENT SAFETY ALERT: Smoke or Gas detected in the nursery room!"

    try:
        bot.send_message(chat_id, text)
    except Exception as e:
        print("Telegram send failed:", e)


def send_hungry_alert():
    text = "👶 Baby is hungry!"

    try:
        bot.send_message(chat_id, text)
    except Exception as e:
        print("Telegram send failed:", e)


def send_tired_alert():
    text = "👶 Baby is tired! ⚠️ Attention required."

    try:
        bot.send_message(chat_id, text)
    except Exception as e:
        print("Telegram send failed:", e)
