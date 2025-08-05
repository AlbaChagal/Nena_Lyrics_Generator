from typing import Optional, Union, Type

Number: Type = Union[int, float]


class ConfusionMatrix(object):
    __slots__ = ('tp', 'fp', 'fn', 'tn')
    def __init__(self, tp: Number = 0, tn: Number = 0, fp: Number = 0, fn: Number = 0) -> None:

        self.tp: Number = tp
        self.tn: Number = tn
        self.fp: Number = fp
        self.fn: Number = fn

    def __add__(self, other: 'ConfusionMatrix') -> 'ConfusionMatrix':
        out = ConfusionMatrix()
        for slot in other.__slots__:
            setattr(out, slot, getattr(self, slot) + getattr(other, slot))
        return out

    def __truediv__(self, other: Number) -> 'ConfusionMatrix':
        out = ConfusionMatrix()
        for slot in self.__slots__:
            setattr(out, slot, getattr(self, slot) / other)
        return out



class Metrics(object):
    __slots__ = (
        'tpr', 'tnr', 'acc', 'confusion_matrix'
    )
    def __init__(self,
                 tpr: Number = 0.,
                 tnr: Number = 0.,
                 acc: Number = 0.,
                 confusion_matrix: ConfusionMatrix = ConfusionMatrix()) -> None:

        self.tpr: Number = tpr
        self.tnr: Number = tnr
        self.acc: Number = acc
        self.confusion_matrix: ConfusionMatrix = confusion_matrix

    def __add__(self, other: 'Metrics') -> 'Metrics':
        out: 'Metrics' = Metrics()
        for slot in self.__slots__:
            setattr(out, slot, getattr(self, slot) + getattr(other, slot))
        return out

    def __truediv__(self, other: Number) -> 'Metrics':
        out: 'Metrics' = Metrics()
        for slot in self.__slots__:
            setattr(out, slot, getattr(self, slot) / other)
        return out

    def __str__(self) -> str:
        out: str = ''
        for slot in self.__slots__:
            out += f'slot {slot}: {str(getattr(self, slot))}, '
        return out[:-2]



class AllMetrics(object):
    __slots__ = (
        'general_token_metrics',
        'eos_token_metrics',
        'new_line_token_metrics',
        'separator_token_metrics',
        'bos_token_metrics',
        'unknown_token_metrics'
    )
    def __init__(
            self,
            general_token_metrics: Metrics,
            eos_token_metrics: Metrics,
            new_line_token_metrics: Metrics,
            separator_token_metrics: Metrics,
            bos_token_metrics: Metrics,
            unknown_token_metrics: Metrics
    ) -> None:

        self.general_token_metrics: Metrics = general_token_metrics
        self.eos_token_metrics: Metrics = eos_token_metrics
        self.new_line_token_metrics: Metrics = new_line_token_metrics
        self.separator_token_metrics: Metrics = separator_token_metrics
        self.bos_token_metrics: Metrics = bos_token_metrics
        self.unknown_token_metrics: Metrics = unknown_token_metrics

    @staticmethod
    def init_empty_instance() -> 'AllMetrics':
        return AllMetrics(
            general_token_metrics=Metrics(),
            eos_token_metrics=Metrics(),
            new_line_token_metrics=Metrics(),
            separator_token_metrics=Metrics(),
            bos_token_metrics=Metrics(),
            unknown_token_metrics=Metrics()
        )

    def __add__(self, other: 'AllMetrics') -> 'AllMetrics':
        return AllMetrics(
            general_token_metrics=self.general_token_metrics + other.general_token_metrics,
            eos_token_metrics=self.eos_token_metrics + other.eos_token_metrics,
            new_line_token_metrics=self.new_line_token_metrics + other.new_line_token_metrics,
            separator_token_metrics=self.separator_token_metrics + other.separator_token_metrics,
            bos_token_metrics=self.bos_token_metrics + other.bos_token_metrics,
            unknown_token_metrics=self.unknown_token_metrics + other.unknown_token_metrics,
        )

    def __truediv__(self, other: Number) -> 'AllMetrics':
        return AllMetrics(
            general_token_metrics=self.general_token_metrics / other,
            eos_token_metrics=self.eos_token_metrics / other,
            new_line_token_metrics=self.new_line_token_metrics / other,
            separator_token_metrics=self.separator_token_metrics / other,
            bos_token_metrics=self.bos_token_metrics / other,
            unknown_token_metrics=self.unknown_token_metrics / other,
        )
