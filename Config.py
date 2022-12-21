# бывший SR
part_frames = 16_000 * 3

# Параметры модели
n_fft = 1024  # размер STFT свертки
hop_length = n_fft // 4  # параметр для STFT
win_length = n_fft  # размер окна STFT
window = 'hann_window'  # тип окна STFT (еще есть 'hamm_window')
size = n_fft // 2  # размер после линейного преобразования в Conformer
conf_blocks_num = 6  # Количество ConformerBlock'ов в Conformer'е
conv_kernel_size = 31  # kernel_size для ConvModule
