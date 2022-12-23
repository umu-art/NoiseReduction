from torch import nn

from model_D.layers.ConformerBlock import ConformerBlock

from model_D.layers.GlobalLayerNorm import GlobalLayerNorm


class Conformer(nn.Module):
    """
        Аргументы:
            size: размер на выходе линейного преобразования
            conf_blocks_num: количество ConformerBlock'ов
            conv_kernel_size: параметр для DepthWise слоя
            num_heads, expansion_factor, dropout_conv, dropout_multi_head, dropout_feed_forward: параметры для ConformerBlock (см. ConformerBlock.py)
        Возвращает:
            Объект типа Conformer (основная часть модели)
    """

    def __init__(self, size: int, conf_blocks_num: int,
                 conv_kernel_size: int, num_heads: int = 4, expansion_factor: int = 4,
                 dropout_conv: float = 0.1, dropout_multi_head: float = 0.1, dropout_feed_forward: float = 0.1):
        super().__init__()
        self.size = size
        self.conf_blocks_size = conf_blocks_num
        self.layer_norm = GlobalLayerNorm(size)
        self.conf_blocks = nn.ModuleList(
            [ConformerBlock(size=size, num_heads=num_heads,
                            conv_kernel_size=conv_kernel_size, expansion_factor=expansion_factor,
                            dropout_conv=dropout_conv, dropout_multi_head=dropout_multi_head,
                            dropout_feed_forward=dropout_feed_forward)
             for _ in
             range(conf_blocks_num)])
        self.lin_second = nn.Linear(size, size)

    """
        Аргументы: 
            x (torch.tensor): входные данные
        Возвращает:
            out (torch.tensor): данный массив, прогнанный через все нужные слои
    """

    def forward(self, x):
        x = self.layer_norm(x)
        for block in self.conf_blocks:
            x = block(x)
        x = self.lin_second(x)
        return x
