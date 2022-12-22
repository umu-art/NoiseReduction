import Token
from AudioMetods import save_audio
from model.NR_Model import NRModel
import telebot
import subprocess

bot = telebot.TeleBot(Token.token)
neiro = NRModel()


def work(abstract, chat_id):
    file_info = bot.get_file(abstract.file_id)
    file_type = abstract.mime_type[6:]

    downloaded_file = bot.download_file(file_info.file_path)

    print(f'New audio from {chat_id}')

    bot.send_message(chat_id, 'Принял, работаю...')

    file_name = f'audio_from_{chat_id}'

    src = f'cache/{file_name}.{file_type}'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    subprocess.run(['ffmpeg/bin/ffmpeg.exe', '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', src, f'cache/{file_name}_conv.wav'])

    wave = neiro(f'cache/{file_name}_conv.wav')
    save_audio(f'cache/{file_name}_clean.wav', wave)

    print(f'Successful for {chat_id}\n')

    audio = open(f'cache/{file_name}_clean.wav', 'rb')
    bot.send_audio(chat_id, audio, title='Ваша запись без шума')
    audio.close()


@bot.message_handler(commands=['start', 'help'])
def hello(message):
    bot.send_message(message.chat.id, 'Привет, я бот NR\nОтправь мне аудиозапись или голосовое сообщение,\n'
                                      'и я верну его же без шумов')


@bot.message_handler(content_types=["audio"])
def get_audio(message):
    work(message.audio, message.chat.id)


@bot.message_handler(content_types=["document"])
def get_document(message):
    work(message.document, message.chat.id)


@bot.message_handler(content_types=["voice"])
def get_voice(message):
    work(message.voice, message.chat.id)


print('Started')
bot.infinity_polling()
