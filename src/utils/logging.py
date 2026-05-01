import sys
import torch


class FilteredStderr:
    # drops the harmless AssertionError spam from DataLoader worker cleanup
    _NOISE = (
        "Exception ignored", "_MultiProcessingDataLoaderIter",
        "_shutdown_workers", "is_alive", "can only test a child process",
        "AssertionError", "dataloader.py", "__del__",
    )

    def __init__(self, real):
        self.real = real

    def write(self, s):
        if not any(m in s for m in self._NOISE):
            self.real.write(s)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


def install_stderr_filter():
    if not isinstance(sys.stderr, FilteredStderr):
        sys.stderr = FilteredStderr(sys.stderr)
