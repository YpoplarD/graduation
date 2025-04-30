import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

class CLIPReID(nn.Module):
    def __init__(self, clip_model_name='ViT-B/16', num_classes=0, num_features=512, 
                 dropout=0.5, cut_at_pooling=False, fp16=False, **kwargs):
        """
        Args:
            clip_model_name: CLIP模型名称 (ViT-B/16, ViT-L/14等)
            num_classes: 分类数量 (0表示只提取特征)
            num_features: 输出特征维度
            dropout: Dropout概率
            cut_at_pooling: 是否跳过特征头
            fp16: 是否使用半精度
        """
        super(CLIPReID, self).__init__()
        

        self.clip_model, _ = clip.load(clip_model_name)
        self.visual = self.clip_model.visual
        

        self._setup_parameter_freezing()
        

        self.num_features = num_features
        self.feature_head = self._build_feature_head(dropout) if not cut_at_pooling else None
        

        self.classifier = nn.Linear(num_features, num_classes) if num_classes > 0 else None
        if self.classifier is not None:
            kaiming_normal_(self.classifier.weight, mode='fan_out')
            nn.init.constant_(self.classifier.bias, 0)
        

        if not fp16:
            self.float()

    def _setup_parameter_freezing(self):
        """解冻最后3个Transformer block和投影层"""
        total_layers = len(self.visual.transformer.resblocks)
        print(f"CLIP模型总层数: {total_layers}")  # ViT-B16应为12
        
        freeze_count = 0
        trainable_count = 0
        
        # 解冻策略
        for name, param in self.visual.named_parameters():

            if any(f'transformer.resblocks.{i}.' in name for i in range(total_layers-3, total_layers)):
                param.requires_grad = True
                trainable_count += 1

            elif 'proj' in name or 'ln_post' in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                freeze_count += 1
        

        for name, param in self.named_parameters():
            if 'feature_head' in name or 'classifier' in name:
                param.requires_grad = True
                trainable_count += 1
        
        print(f"冻结参数层: {freeze_count}, 可训练层: {trainable_count}")

    def _build_feature_head(self, dropout):
        """构建特征提取头"""
        return nn.Sequential(
            nn.Linear(self.visual.output_dim, self.num_features * 2),
            nn.BatchNorm1d(self.num_features * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features * 2, self.num_features),
            nn.BatchNorm1d(self.num_features)
        )

    def forward(self, x):

        if x.dtype != next(self.parameters()).dtype:
            x = x.type_as(next(self.parameters()))
        

        x = self.visual(x)
        

        if self.feature_head is not None:
            x = self.feature_head(x)
        

        if not self.training:
            return F.normalize(x, p=2, dim=1)
        
  
        if self.classifier is not None:
            return F.normalize(x, p=2, dim=1), self.classifier(x)
        return F.normalize(x, p=2, dim=1)

    def float(self):
        super().float()
        self.clip_model.float()
        return self