from abc import ABC, abstractmethod


class PipelineAction(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Pipeline:
    def __init__(self, items: list):
        self._items = items

    def __call__(self, *args, **kwargs):
        res = self._items[0](*args, **kwargs)
        for i in range(1, len(self._items)):
            res = self._items[i](res)

        return res
