
# бывший SR
part_frames = 16_000 * 3

# Параметры Dataset
prefix_root = 'E:/'  # начало ссылок на входные данные
clean_speech_data_root = f'{prefix_root}LibriSpeech/train-clean-100'  # местоположение директории с чистыми записями для обучения
noise_root = f'{prefix_root}musan/noise'  # местоположение директории с шумами для обучения
noise_eval_root = f'{prefix_root}DEMAND'  # местоположение директории с шумами для теста
clean_pattern = f'{clean_speech_data_root}/**/*.flac'  # паттерн названия файла с записью диктора
noise_pattern = f'{noise_root}/**/*.wav'  # паттерн названия файла с шумом для обучения
noise_eval_pattern = f'{noise_eval_root}/**/*.wav'  # паттерн названия файла с шумом для теста
cache_folder = 'cache/'  # путь до директории с кешем
snr_range = (2, 5)  # разброс snr для обучения [left..right]
chance_same_gender = 0.2

# Параметры модели
n_fft = 1024  # размер STFT свертки
hop_length = n_fft // 4  # параметр для STFT
win_length = n_fft  # размер окна STFT
window = 'hann_window'  # тип окна STFT (еще есть 'hamm_window')
size = n_fft // 2  # размер после линейного преобразования в Conformer
conf_blocks_num = 12  # Количество ConformerBlock'ов в Conformer'е
conv_kernel_size = 31  # kernel_size для ConvModule
w_len = 40

# Параметры обучения
batch_size = 4  # размер выборки для одной итерации по датасету
iters_per_epoch = 10  # количество итераций на эпоху обучения
epochs = 2  # количество эпох обучения
clip_val = 5  # параметр для изменения learning_rate
save_path = 'out.tar'  # путь до архива с сохраненными данными после обучения/тестирования

# Параметры scheduler
start_lr = 1e-5  # стартовая learning_rate
warmup_epochs = 10  # количесто wapmup_epoch
step_size = 10  # размер шага
gamma = 0.99  # быстрота изменения learning_rate (можно покрутить для улучшения обучения)
min_lr = 1e-5  # искусственный минимальный learning_rate для обучения
opt_lr = 1e-3 # оптимальный уровень learning_rate для обучения

