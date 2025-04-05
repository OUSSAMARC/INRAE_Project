import torchvision.models as models
import torch.nn as nn
import segmentation_models_pytorch as smp  # pip install segmentation-models-pytorch


class SegmentationModelBuilder:
    def __init__(self, backbone_name="resnet34", num_classes=3, encoder_weights="imagenet", arch="Unet"):
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.encoder_weights = encoder_weights
        self.arch = arch.lower()

    def build(self):
        if self.arch == "unet":
            return smp.Unet(
                encoder_name=self.backbone_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
            )

        elif self.arch == "deeplabv3":
            return smp.DeepLabV3(
                encoder_name=self.backbone_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
            )

        elif self.arch == "fpn":
            return smp.FPN(
                encoder_name=self.backbone_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=self.num_classes,
            )

        else:
            raise ValueError(f"Unknown architecture: {self.arch}")
