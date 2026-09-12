import telebot
from config import API_TOKEN

bot = telebot.TeleBot(API_TOKEN)

CHAT_ID = 123456789


def send_message(message):
    bot.send_message(CHAT_ID, message)


def send_gas_alert():
    send_message(
        "🚨 SAFETY ALERT!\n"
        "Gas or smoke has been detected in the nursery."
    )


def send_hungry_alert():
    send_message(
        "👶 Baby Cry Detected\n"
        "Classification: Hungry"
    )


def send_tired_alert():
    send_message(
        "👶 Baby Cry Detected\n"
        "Classification: Tired\n"
        "⚠️ Attention required."
    )


def send_temperature_alert(temp):
    send_message(
        f"🌡️ HIGH TEMPERATURE ALERT!\n"
        f"Nursery temperature: {temp}°C"
    )
