from pathlib import Path

from AudioMetods import save_audio, read_audio
from model.NR_Model import NRModel
import telebot

bot = telebot.TeleBot("5659090384:AAHnJt9t6UWU2bEIg2uF8mmWB_YTn7_1Hi4")
neiro = NRModel()


def work(abstract, chat_id):
    file_info = bot.get_file(abstract.file_id)
    file_type = abstract.mime_type[6:]
    downloaded_file = bot.download_file(file_info.file_path)

    print(f'New audio from {chat_id}')

    bot.send_message(chat_id, 'Принял, работаю...')

    src = 'cache/in.' + file_type
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    wave = neiro(src)
    save_audio('cache/clean.wav', wave)

    print(f'Successful for {chat_id}\n')

    audio = open('cache/clean.wav', 'rb')
    bot.send_audio(chat_id, audio)
    audio.close()


@bot.message_handler(content_types=["audio"])
def get_audio(message):
    work(message.audio, message.chat.id)


@bot.message_handler(content_types=["document"])
def get_audio(message):
    work(message.document, message.chat.id)


@bot.message_handler(content_types=["voice"])
def get_audio(message):
    work(message.voice, message.chat.id)


print('Started')
bot.infinity_polling()
