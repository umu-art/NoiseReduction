# бывший SR
part_frames = 16_000

# Параметры Dataset
clean_speech_data_root = 'E:/LibriSpeech/train-clean-100'
noise_root = 'E:/musan/noise'
clean_pattern = f'{clean_speech_data_root}/**/*.flac'
noise_pattern = f'{noise_root}/**/*.wav'
cache_folder = 'cache/'
snr_range = (2, 10)

# Параметры модели
n_fft = 1024
hop_length = n_fft // 4
win_length = n_fft
window = 'hann_window'
size = n_fft // 2
conf_blocks_num = 12
conv_kernel_size = 31

# Параметры обучения
betas = (0.9, 0.999)
lr = 1e-4
batch_size = 4
iters_per_epoch = 10
epochs = 15
save_path = 'out.tar'
