from pathlib import Path

import librosa
import torch

import Token
from AudioMetods import save_audio, read_audio
from model import Config
from model.Conformer import Conformer
from model.NR_Model import NRModel
import telebot

model = Conformer(Config.n_fft, Config.hop_length, Config.win_length, Config.window,
                          Config.size, Config.conf_blocks_num, Config.conv_kernel_size)

snap = torch.load('model/model.tar', map_location='cpu')
model_state_dict = snap['model']
model.load_state_dict(model_state_dict)
model.eval()

mix, sr, d = read_audio('cache/dd.wav')
mix = mix[:, 0]
mix = librosa.resample(mix, orig_sr=sr, target_sr=16_000)
audio = torch.from_numpy(mix)[None]
ans = model(audio)[0]
save_audio('out.wav', ans.detach().numpy())

# bot = telebot.TeleBot(Token.token)
# neiro = NRModel()
#
# wave = neiro('cache/dd.wav')
# save_audio('cache/clean.wav', wave)


def work(abstract, chat_id):
    file_info = bot.get_file(abstract.file_id)
    file_type = abstract.mime_type[6:]

    if file_type == 'ogg':
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
