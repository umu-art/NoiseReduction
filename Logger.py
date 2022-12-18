import tensorboard.plugins.core.core_plugin
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

tensorboard.plugins.core.core_plugin.DEFAULT_PORT = 1771

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'log/'])
url = tb.launch()
print(f"Tensorflow listening on {url}")

writer = SummaryWriter("log")


def write_point(t: str, x: int, cur_snr, inp_snr, loss):
    writer.add_scalars(t + '_snrs', {'cur_snr': cur_snr,
                              'inp_snr': inp_snr}, x)
    writer.add_scalar(t + '_loss', loss, x)
    writer.flush()
