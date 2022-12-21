import argparse
import tensorboard.plugins.core.core_plugin
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

tensorboard.plugins.core.core_plugin.DEFAULT_PORT = 1771

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log',
    type=str,
    default=datetime.now().strftime('%m.%d_%H.%M'),
    help='folder to save logs'
)
namespace = parser.parse_args()

writer = SummaryWriter("log/" + namespace.log)


def write_point(t: str, x: int, loss):
    writer.add_scalars('loss', {t: loss}, x)
    writer.flush()


def save_audio(audio, name):
    writer.add_audio(name, audio, sample_rate=16000)


def write_epoch_point(t: str, x: int, loss):
    writer.add_scalars('epoch_loss', {t: loss}, x)
    writer.flush()


def write_lr(lr, x):
    writer.add_scalars('lr', {'.': lr[0]}, x)
    writer.flush()


def write_grad_norm(grad_norm, x):
    writer.add_scalars('grad_norm', {'.': grad_norm}, x)
    writer.flush()
