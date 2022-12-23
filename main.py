import Token
from AudioMetods import save_audio
from model_BSS.BSS_Model import BSSModel
from model_D.D_Model import DModel
from model_NR.NR_Model import NRModel
import telebot
import subprocess
from telebot import types

bot = telebot.TeleBot(Token.token)
neiro = NRModel()
spliter = BSSModel()
detect = DModel()
mode = {}

ffmpeg = 'ffmpeg/bin/ffmpeg.exe'


def work(abstract, chat_id):
    if chat_id not in mode.keys():
        mode[chat_id] = 0
    file_info = bot.get_file(abstract.file_id)
    file_type = abstract.mime_type[6:]

    downloaded_file = bot.download_file(file_info.file_path)

    print(f'New audio from {chat_id}')

    bot.send_message(chat_id, 'Принял, работаю...')

    file_name = f'audio_from_{chat_id}'

    src = f'cache/{file_name}.{file_type}'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', src, f'cache/{file_name}_conv.wav'])

    two_speakers = detect(f'cache/{file_name}_conv.wav')

    if (mode[chat_id] == 0 and not two_speakers) or (mode[chat_id] == 1):
        print('Режим шумодава')
        wave = neiro(f'cache/{file_name}_conv.wav')
        save_audio(f'cache/{file_name}_clean.wav', wave)

        audio = open(f'cache/{file_name}_clean.wav', 'rb')
        bot.send_audio(chat_id, audio, title='Ваша запись без шума', duration=len(wave) // 16_000, performer='NR_Bot')
        audio.close()

    if (mode[chat_id] == 0 and two_speakers) or (mode[chat_id] == 2):
        print('Режим разделения')
        wave = spliter(f'cache/{file_name}_conv.wav')
        save_audio(f'cache/{file_name}_split0.wav', wave[0])
        save_audio(f'cache/{file_name}_split1.wav', wave[1])

        wave = neiro(f'cache/{file_name}_split0.wav')
        save_audio(f'cache/{file_name}_split_cl0.wav', wave)

        wave = neiro(f'cache/{file_name}_split1.wav')
        save_audio(f'cache/{file_name}_split_cl1.wav', wave)

        audio = open(f'cache/{file_name}_split_cl0.wav', 'rb')
        bot.send_audio(chat_id, audio, title='Первый спикер', duration=len(wave) // 16_000, performer='NR_Bot')
        audio = open(f'cache/{file_name}_split_cl1.wav', 'rb')
        bot.send_audio(chat_id, audio, title='Второй спикер', duration=len(wave) // 16_000, performer='NR_Bot')

    print(f'Successful for {chat_id}\n')


@bot.message_handler(commands=['set_mode'])
def hello(message):
    mm = types.InlineKeyboardMarkup(row_width=1)
    button1 = types.InlineKeyboardButton("Режим: шумоподавление", callback_data='1')
    button2 = types.InlineKeyboardButton("Режим: разделение речи", callback_data='2')
    button3 = types.InlineKeyboardButton("Режим: автоматически", callback_data='0')
    mm.add(button1, button2, button3)
    bot.send_message(message.chat.id, 'Режимы:\n' +
                     'Шумоподавление - бот будет шум подавлять\n' +
                     'Разделение речи - бот будет две речи разделять\n' +
                     'Автоматически - бот вернётся в автоматический режим\n',
                     reply_markup=mm)


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:
            if call.data == "0":
                mode[call.message.chat.id] = 0
                bot.answer_callback_query(call.id, 'Бот теперь в автоматическом режиме')
            if call.data == "1":
                mode[call.message.chat.id] = 1
                bot.answer_callback_query(call.id, 'Бот теперь в режиме шумоподавления')
            if call.data == "2":
                mode[call.message.chat.id] = 2
                bot.answer_callback_query(call.id, 'Бот теперь в режиме разделения речи')
    except Exception as e:
        print(repr(e))


@bot.message_handler(commands=['start', 'help'])
def hello(message):
    if message.chat.id not in mode.keys():
        mode[message.chat.id] = 0
    bot.send_photo(message.chat.id, open('hi.jpg', 'rb'),
                   'Налево пойдёшь, шум из аудио уберешь. Направо пойдёшь, запись двух человек разделишь. Прямо '
                   'пойдёшь, смерть свою сыщешь. Дорогу выберем мы, тебе нужно просто отправить '
                   'голосовое/аудиофайл/кружочек c видео.\n' +
                   'Для слабонервных: /set_mode')


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
    if message.chat.id not in mode.keys():
        mode[message.chat.id] = 0
    file_info = bot.get_file(message.video_note.file_id)
    file_type = 'mp4'
    downloaded_file = bot.download_file(file_info.file_path)

    print(f'New video_note from {message.chat.id}')

    bot.send_message(message.chat.id, 'Принял, работаю...')

    file_name = f'round_from_{message.chat.id}'
    src = f'cache/{file_name}.{file_type}'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', src, '-vn', '-acodec', 'copy', f'cache/{file_name}_extracted.aac'])

    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', f'cache/{file_name}_extracted.aac', f'cache/{file_name}_conv.wav'])

    two_speakers = detect(f'cache/{file_name}_conv.wav')

    if (mode[message.chat.id] == 0 and not two_speakers) or (mode[message.chat.id] == 1):
        print('Режим шумодава')
        wave = neiro(f'cache/{file_name}_conv.wav')
        save_audio(f'cache/{file_name}_clean.wav', wave)
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', src, '-i', f'cache/{file_name}_clean.wav', '-map', '0:v', '-map', '1:a',
                        '-shortest', f'cache/{file_name}_out.mp4'])
        v_note = open(f'cache/{file_name}_out.mp4', 'rb')
        bot.send_video_note(message.chat.id, v_note)
        v_note.close()
    if (mode[message.chat.id] == 0 and two_speakers) or (mode[message.chat.id] == 2):
        print('Режим разделения')
        wave = spliter(f'cache/{file_name}_conv.wav')
        save_audio(f'cache/{file_name}_split0.wav', wave[0])
        save_audio(f'cache/{file_name}_split1.wav', wave[1])

        wave = neiro(f'cache/{file_name}_split0.wav')
        save_audio(f'cache/{file_name}_split_cl0.wav', wave)

        wave = neiro(f'cache/{file_name}_split1.wav')
        save_audio(f'cache/{file_name}_split_cl1.wav', wave)

        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', src, '-i', f'cache/{file_name}_split_cl0.wav', '-map', '0:v', '-map', '1:a',
                        '-shortest', f'cache/{file_name}_out0.mp4'])

        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', src, '-i', f'cache/{file_name}_split_cl1.wav', '-map', '0:v', '-map', '1:a',
                        '-shortest', f'cache/{file_name}_out1.mp4'])

        v_note = open(f'cache/{file_name}_out0.mp4', 'rb')
        bot.send_video_note(message.chat.id, v_note)
        v_note.close()

        v_note = open(f'cache/{file_name}_out1.mp4', 'rb')
        bot.send_video_note(message.chat.id, v_note)
        v_note.close()

    print(f'Successful for {message.chat.id}\n')


@bot.message_handler(content_types=['video'])
def get_video_note(message):
    if message.chat.id not in mode.keys():
        mode[message.chat.id] = 0
    file_info = bot.get_file(message.video.file_id)
    file_type = message.video.mime_type[6:]
    downloaded_file = bot.download_file(file_info.file_path)

    print(f'New video from {message.chat.id}')

    bot.send_message(message.chat.id, 'Принял, работаю...')

    file_name = f'video_from_{message.chat.id}'
    src = f'cache/{file_name}.{file_type}'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', src, '-vn', '-acodec', 'copy', f'cache/{file_name}_extracted.aac'])

    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                    '-y', '-i', f'cache/{file_name}_extracted.aac', f'cache/{file_name}_conv.wav'])

    two_speakers = detect(f'cache/{file_name}_conv.wav')

    if (mode[message.chat.id] == 0 and not two_speakers) or (mode[message.chat.id] == 1):
        print('Режим шумодава')
        wave = neiro(f'cache/{file_name}_conv.wav')
        save_audio(f'cache/{file_name}_clean.wav', wave)
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', src, '-i', f'cache/{file_name}_clean.wav', '-map', '0:v', '-map', '1:a',
                        '-shortest', f'cache/{file_name}_out.{file_type}'])
        v_note = open(f'cache/{file_name}_out.{file_type}', 'rb')
        bot.send_video(message.chat.id, v_note)
        v_note.close()
    if (mode[message.chat.id] == 0 and two_speakers) or (mode[message.chat.id] == 2):
        print('Режим разделения')
        wave = spliter(f'cache/{file_name}_conv.wav')
        save_audio(f'cache/{file_name}_split0.wav', wave[0])
        save_audio(f'cache/{file_name}_split1.wav', wave[1])

        wave = neiro(f'cache/{file_name}_split0.wav')
        save_audio(f'cache/{file_name}_split_cl0.wav', wave)

        wave = neiro(f'cache/{file_name}_split1.wav')
        save_audio(f'cache/{file_name}_split_cl1.wav', wave)

        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', src, '-i', f'cache/{file_name}_split_cl0.wav', '-map', '0:v', '-map', '1:a',
                        '-shortest', f'cache/{file_name}_out0.{file_type}'])

        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                        '-y', '-i', src, '-i', f'cache/{file_name}_split_cl1.wav', '-map', '0:v', '-map', '1:a',
                        '-shortest', f'cache/{file_name}_out1.{file_type}'])

        v_note = open(f'cache/{file_name}_out0.{file_type}', 'rb')
        bot.send_video(message.chat.id, v_note)
        v_note.close()

        v_note = open(f'cache/{file_name}_out1.{file_type}', 'rb')
        bot.send_video(message.chat.id, v_note)
        v_note.close()

    print(f'Successful for {message.chat.id}\n')


print('Started')
bot.infinity_polling()
