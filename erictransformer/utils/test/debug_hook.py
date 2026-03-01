import contextlib
import typing

class DebugHook:
    hook: typing.Callable | None

    def __init__(self) -> None:
        self.hook = None

    def __call__(self, *a, **k) -> typing.Any:
        if self.hook:
            return self.hook(*a,**k)

    @contextlib.contextmanager
    def set(self, hook):
        self.hook = hook
        try:
            yield
        finally:
            self.hook = None
