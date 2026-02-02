from abc import ABC, abstractmethod


class EricValidator(ABC):
    def __init__(self):
        self.validate_init()

    @abstractmethod
    def validate_init(self):
        pass
