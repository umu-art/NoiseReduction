import telebot

bot = telebot.TeleBot("TOKEN")


@bot.message_handler(content_types=["audio"])
def try_to_get(message):
    file_info = bot.get_file(message.audio.file_id)
    file_type = message.audio.mime_type[6:]
    downloaded_file = bot.download_file(file_info.file_path)

    src = 'data/' + message.audio.file_unique_id + '.' + file_type
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    audio = open(f"data/{message.audio.file_unique_id}.{file_type}", 'rb')
    bot.send_audio(message.chat.id, audio)
    audio.close()


bot.infinity_polling()

