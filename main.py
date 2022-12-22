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
    bot.send_photo(message.chat.id, open('hi.jpg', 'rb'), 'Налево пойдёшь, шум из аудио уберешь. Направо пойдёшь, запись двух человек '
                                      'разделишь. Прямо пойдёшь, смерть свою сыщешь. Дорогу выберем мы, тебе нужно '
                                      'просто отправить голосовое/аудиофайл/кружочек c видео.')


@bot.message_handler(content_types=["audio"])
def get_audio(message):
    work(message.audio, message.chat.id)


@bot.message_handler(content_types=["document"])
def get_document(message):
    work(message.document, message.chat.id)


@bot.message_handler(content_types=["voice"])
def get_voice(message):
    work(message.voice, message.chat.id)


@bot.message_handler(content_types=['video_note'])
def get_video_note(message):
    file_info = bot.get_file(message.video_note.file_id)
    file_type = 'mp4'
    downloaded_file = bot.download_file(file_info.file_path)

    print(f'New video_note from {message.chat.id}')

    bot.send_message(message.chat.id, 'Принял, работаю...')

    file_name = f'round_from_{message.chat.id}'
    src = f'cache/{file_name}.{file_type}'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    subprocess.run(['ffmpeg/bin/ffmpeg.exe', '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', src, '-vn', '-acodec', 'copy', f'cache/{file_name}_extracted.aac'])

    subprocess.run(['ffmpeg/bin/ffmpeg.exe', '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', f'cache/{file_name}_extracted.aac', f'cache/{file_name}_conv.wav'])

    wave = neiro(f'cache/{file_name}_conv.wav')
    save_audio(f'cache/{file_name}_clean.wav', wave)

    subprocess.run(['ffmpeg/bin/ffmpeg.exe', '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', src, '-i',  f'cache/{file_name}_clean.wav', '-map', '0:v', '-map', '1:a',
                    '-shortest', f'cache/{file_name}_out.mp4'])

    print(f'Successful for {message.chat.id}\n')

    v_note = open(f'cache/{file_name}_out.mp4', 'rb')
    bot.send_video_note(message.chat.id, v_note)
    v_note.close()


print('Started')
bot.infinity_polling()
