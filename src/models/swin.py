import timm

from .efficientnet import TimmBinaryWrapper


def build_swin_t(pretrained=True, drop_rate=0.1, drop_path_rate=0.2):
    base = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                              num_classes=1, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
    backbone, head = [], []
    for name, p in base.named_parameters():
        (head if "head" in name else backbone).append(p)
    return TimmBinaryWrapper(base), backbone, head
