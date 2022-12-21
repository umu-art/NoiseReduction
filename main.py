import Token
from AudioMetods import save_audio
from model.NR_Model import NRModel
import telebot

bot = telebot.TeleBot(Token.token)
neiro = NRModel()


def work(abstract, chat_id):
    file_info = bot.get_file(abstract.file_id)
    # file_type = abstract.mime_type[6:]
    file_type = 'wav'

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
