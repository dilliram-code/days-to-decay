"""Deep Learning Regression Model Architectures for Days to Decay."""
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("regression_model")


class FoodAgingRegressor(nn.Module):
    """
    Continuous-value regression network for estimating remaining perishable shelf-life (days).
    
    Architecture:
      Pretrained Feature Backbone (e.g. ResNet50 or EfficientNet-B0)
      -> Global Adaptive Average Pooling
      -> Flatten
      -> Linear projection (feature_dim -> hidden_dim)
      -> Batch Normalization & LeakyReLU
      -> Dropout
      -> Linear output layer (hidden_dim -> 1)
      -> ReLU (enforces physical constraint: remaining days >= 0.0)
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        activation: str = "relu",
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower().strip()
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation_name = activation.lower().strip()

        # 1. Instantiate backbone
        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            # Remove original classification head
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported backbone: '{backbone_name}'. Choose 'resnet50', 'resnet18', or 'efficientnet_b0'.")

        # 2. Continuous regression head
        head_layers = [
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        ]

        if self.activation_name == "relu":
            head_layers.append(nn.ReLU())
        elif self.activation_name == "softplus":
            head_layers.append(nn.Softplus())
        elif self.activation_name == "none":
            pass
        else:
            raise ValueError(f"Unsupported output activation: {activation}")

        self.regression_head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features)
        pred_days = self.regression_head(pooled)
        return pred_days

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freezes or unfreezes all parameters within the feature extractor backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        logger.info(f"Backbone {'frozen' if freeze else 'unfrozen'}.")

    def set_fine_tuning_layers(self, unfreeze_last_n_blocks: int = 1) -> None:
        """
        Freezes early feature layers while permitting high-level residual blocks
        to fine-tune aging-specific texture and browning patterns.
        """
        # Freeze entire backbone first
        self.freeze_backbone(True)

        if self.backbone_name.startswith("resnet"):
            # ResNet children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
            # Children list length is 8
            layers_to_unfreeze = list(self.backbone.children())[-unfreeze_last_n_blocks:]
            for block in layers_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze last {unfreeze_last_n_blocks} residual stages of {self.backbone_name}.")
        else:
            # EfficientNet: unfreeze top feature blocks
            num_blocks = len(self.backbone)
            for idx in range(max(0, num_blocks - unfreeze_last_n_blocks * 2), num_blocks):
                for param in self.backbone[idx].parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze top feature stages of {self.backbone_name}.")

        # Head always trainable
        for param in self.regression_head.parameters():
            param.requires_grad = True
