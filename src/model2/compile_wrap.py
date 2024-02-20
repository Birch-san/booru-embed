# by Katherine Crowson
# https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/models/flags.py#L34
from functools import update_wrapper
from os import environ
import torch

def get_use_compile() -> bool:
    return environ.get("BOORU_USE_COMPILE", "1") == "1"

class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        if get_use_compile():
            try:
                self._compiled_function = torch.compile(self.function, *self.args, **self.kwargs)
            except RuntimeError:
                self._compiled_function = self.function
        else:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)