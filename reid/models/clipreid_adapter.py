import torch.nn as nn

class CLIPReIDAdapter(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        # 直接使用CLIPReID的视觉编码器，不依赖PCB的backbone
        self.image_encoder = clip_model.image_encoder  
        
        # 添加PCB需要的分类头
        self.classifier = nn.ModuleList([
            nn.Linear(clip_model.feature_dim, num_classes) 
            for _ in range(6)  # PCB默认6个头
        ])
        
    def forward(self, x):
        features = self.image_encoder(x)
        if isinstance(features, tuple):
            features = features[0]  # 取主特征
        return features, [cls(features) for cls in self.classifier]