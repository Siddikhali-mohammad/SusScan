import timm

from .efficientnet import TimmBinaryWrapper


def build_clip_vit_b16(pretrained=True, dropout=0.2):
    # timm's CLIP ViT-B/16 with openai weights; single linear head
    name = "vit_base_patch16_clip_224.openai" if pretrained else "vit_base_patch16_clip_224"
    base = timm.create_model(name, pretrained=pretrained, num_classes=1)
    m = TimmBinaryWrapper(base)
    return m, list(base.parameters()), []
