import soundfile as sf
from Config import part_frames


def read_audio(path: str):
    """
        Ф-я читает аудио
        Вход:
            path (str) - путь к файлу
        Выход:
            **output** - (само аудио, частота, продолжительность)
    """
    audio, samplerate = sf.read(path)
    duration = len(audio) / samplerate
    return audio, samplerate, duration


def save_audio(file, x):
    """
        Ф-я сохраняет аудио
        Вход:
            file (str) - куда сохранить
            x (tensor) - что нужно сохранить
    """
    sf.write(file, x, 16_000)
