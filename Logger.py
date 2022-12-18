import tensorboard.plugins.core.core_plugin
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import Config

tensorboard.plugins.core.core_plugin.DEFAULT_PORT = 1771

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'log/'])
url = tb.launch()
print(f"Tensorflow listening on {url}")

writer = SummaryWriter("log")


def write_point(t: str, x: int, cur_snr, inp_snr, loss):
    writer.add_scalars(t, {'cur_snr': cur_snr,
                              'inp_snr': inp_snr}, x)
    writer.add_scalars('loss', {t: loss}, x)
    writer.flush()


def save_audio(audio, name):
    writer.add_audio(name, audio, sample_rate=Config.part_frames)
