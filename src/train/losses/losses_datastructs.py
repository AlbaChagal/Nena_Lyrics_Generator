from typing import Union


class AllLosses(object):
    __slots__ = (
        'total_loss',
        'general_token_loss',
        'eos_token_loss',
        'new_line_token_loss',
        'unknown_token_loss',
        'separator_token_loss',
        'bos_token_loss'
    )

    def __init__(
            self,
            total_loss: float = 0.,
            general_token_loss: float = 0.,
            eos_token_loss: float = 0.,
            new_line_token_loss: float = 0.,
            unknown_token_loss: float = 0.,
            separator_token_loss: float = 0.,
            bos_token_loss: float = 0.
    ):
        self.total_loss: float = total_loss
        self.general_token_loss: float = general_token_loss
        self.eos_token_loss: float = eos_token_loss
        self.new_line_token_loss: float = new_line_token_loss
        self.unknown_token_loss: float = unknown_token_loss
        self.separator_token_loss: float = separator_token_loss
        self.bos_token_loss: float = bos_token_loss

    def __add__(self, other: 'AllLosses') -> 'AllLosses':
        out: 'AllLosses' = AllLosses()
        for slot in self.__slots__:
            setattr(out, slot, getattr(other, slot))
        return out

    def __truediv__(self, other: Union[int, float]) -> 'AllLosses':
        out: 'AllLosses' = AllLosses()
        for slot in self.__slots__:
            setattr(out, slot, getattr(self, slot) / other)
        return out

    def __str__(self):
        out: str = ''
        for slot in self.__slots__:
            out += f'{slot}: {getattr(self, slot):.4f}, '
        return out