SR = 16000
clean_speech_data_root = '/content/LibriSpeech/train-clean-100/'
noise_root = '/content/musan/noise'

# Параметры модели
n_fft = 1024
hop_length = n_fft // 4
win_length = n_fft
window = 'hann_window'
size = n_fft // 2
conf_blocks_num = 12
conv_kernel_size = 31
