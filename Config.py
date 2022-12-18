# бывший SR
part_duration = 16_000

# Параметры Dataset
clean_speech_data_root = 'E:/LibriSpeech/train-clean-100'
noise_root = 'E:/musan/noise'

clean_pattern = f'{clean_speech_data_root}/**/*.flac'
noise_pattern = f'{noise_root}/**/*.wav'

cache_folder = 'cache/'

# Параметры модели
n_fft = 1024
hop_length = n_fft // 4
win_length = n_fft
window = 'hann_window'
size = n_fft // 2
conf_blocks_num = 12
conv_kernel_size = 31
