import torch
import torch.nn as nn
import torchvision.models as models
import timm

class EfficientNetModel(nn.Module):
    """EfficientNet 기반 문서 분류 모델"""
    
    def __init__(self, model_name: str = "efficientnet-b0", num_classes: int = 17, dropout_rate: float = 0.3):
        super().__init__()
        
        # EfficientNet 백본
        if model_name == "efficientnet-b0":
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        elif model_name == "efficientnet-b1":
            self.backbone = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)
        elif model_name == "efficientnet-b2":
            self.backbone = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
        elif model_name == "efficientnet-b3":
            self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
        elif model_name == "efficientnet-b4":
            self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        elif model_name == "efficientnet-b5":
            self.backbone = timm.create_model('efficientnet_b5', pretrained=True, num_classes=num_classes)
        elif model_name == "efficientnet-b6":
            self.backbone =timm.create_model('tf_efficientnet_b6.aa_in1k', pretrained=True, num_classes=num_classes)
        elif model_name == "efficientnet-b7":
            self.backbone = timm.create_model('efficientnet_b7', pretrained=True, num_classes=num_classes)
        else:
            raise ValueError(f"지원하지 않는 EfficientNet 모델: {model_name}")
        
        # # 마지막 분류 층을 제거
        # in_features = self.backbone.classifier[1].in_features
        
        # # 새로운 분류 헤드
        # self.backbone.classifier = nn.Sequential(
        #     # nn.Dropout(dropout_rate),
        #     nn.Linear(in_features, num_classes)
        # )
    
    def forward(self, x):
        return self.backbone(x) 