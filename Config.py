# бывший SR
part_frames = 16_000 * 3

# Параметры Dataset
prefix_root = 'E:/'
clean_speech_data_root = 'E:/LibriSpeech/train-clean-100'
noise_root = 'E:/musan/noise'
noise_eval_root = 'E:/DEMAND'
clean_pattern = f'{clean_speech_data_root}/**/*.flac'
noise_pattern = f'{noise_root}/**/*.wav'
noise_eval_pattern = f'{noise_eval_root}/**/*.wav'
cache_folder = 'cache/'
snr_range = (2, 5)

# Параметры модели
n_fft = 1024
hop_length = n_fft // 4
win_length = n_fft
window = 'hann_window'
size = n_fft // 2
conf_blocks_num = 12
conv_kernel_size = 31

# Параметры обучения
batch_size = 4
iters_per_epoch = 10
epochs = 2
clip_val = 5
save_path = 'out.tar'

# Параметры scheduler
start_lr = 1e-5
warmup_epochs = 10
step_size = 10
gamma = 0.99
min_lr = 1e-5
opt_lr = 1e-3
