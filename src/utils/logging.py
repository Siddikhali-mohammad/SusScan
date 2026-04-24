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


def print_env():
    import timm, open_clip, albumentations as A
    print("=" * 60)
    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA avail    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"BF16 support  : {torch.cuda.is_bf16_supported()}")
    print(f"timm          : {timm.__version__}")
    print(f"open_clip     : {open_clip.__version__}")
    print(f"albumentations: {A.__version__}")
    print("=" * 60)
