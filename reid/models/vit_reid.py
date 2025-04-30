import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from .vision_transformer import *
import os
from torch.hub import load_state_dict_from_url
DINO_PRETRAINED_URLS = {
    'vit_tiny': 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
    'vit_small': 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
    'vit_base': 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth'
}

class DINOReID(nn.Module):
    def __init__(self, dino_model_name='vit_small', num_classes=0, num_features=512, 
                 dropout=0.5, cut_at_pooling=False, fp16=False, **kwargs):
        """
        Args:
            dino_model_name: DINO模型名称 (vit_tiny, vit_small, vit_base)
            num_classes: 分类数量 (0表示只提取特征)
            num_features: 输出特征维度
            dropout: Dropout概率
            cut_at_pooling: 是否跳过特征头
            fp16: 是否使用半精度
            kwargs: 其他参数(包含pretrained)
        """
        super(DINOReID, self).__init__()
        
        # 从kwargs中获取pretrained参数，默认True
        pretrained = kwargs.get('pretrained', True)
        
        # 初始化DINO模型
        self.dino_model = self._load_dino_model(dino_model_name, pretrained)
        
        # 设置参数冻结
        self._setup_parameter_freezing()
        
        # 特征维度
        self.num_features = num_features
        self.feature_head = self._build_feature_head(dropout) if not cut_at_pooling else None
        
        # 分类器
        self.classifier = nn.Linear(num_features, num_classes) if num_classes > 0 else None
        if self.classifier is not None:
            trunc_normal_(self.classifier.weight, std=.02)
            nn.init.constant_(self.classifier.bias, 0)
        
        # 精度设置
        if not fp16:
            self.float()

    def _load_dino_model(self, model_name, pretrained=True):
        """加载DINO模型，可选加载预训练权重"""
        if model_name == 'vit_tiny':
            model = vit_tiny(patch_size=16, num_classes=0)
        elif model_name == 'vit_small':
            model = vit_small(patch_size=16, num_classes=0)
        elif model_name == 'vit_base':
            model = vit_base(patch_size=16, num_classes=0)
        else:
            raise ValueError(f"Unknown DINO model name: {model_name}")
        
        if pretrained:
            print(f"Loading pretrained DINO weights for {model_name}...")
            url = DINO_PRETRAINED_URLS.get(model_name)
            if not url:
                raise ValueError(f"No pretrained URL found for {model_name}")
            
            state_dict = load_state_dict_from_url(url, progress=True)
            
            # 适配不同的模型结构
            if 'teacher' in state_dict:
                # 处理DINO v2的权重
                state_dict = {k.replace("module.", ""): v for k, v in state_dict['teacher'].items()}
                state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            
            # 加载权重并过滤不匹配的键
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers")
        
        return model

    def _setup_parameter_freezing(self):
        """解冻最后3个Transformer block和投影层"""
        total_layers = len(self.dino_model.blocks)
        print(f"DINO模型总层数: {total_layers}")  # vit_small应为12
        
        freeze_count = 0
        trainable_count = 0
        
        # 解冻策略
        for name, param in self.dino_model.named_parameters():
            # 解冻最后3个Transformer block
            if any(f'blocks.{i}.' in name for i in range(total_layers-3, total_layers)):
                param.requires_grad = True
                trainable_count += 1
            # 解冻分类头和norm层
            elif 'head' in name or 'norm' in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                freeze_count += 1
        
        # 特征头和分类器总是可训练的
        for name, param in self.named_parameters():
            if 'feature_head' in name or 'classifier' in name:
                param.requires_grad = True
                trainable_count += 1
        
        print(f"冻结参数层: {freeze_count}, 可训练层: {trainable_count}")

    def _build_feature_head(self, dropout):
        """构建特征提取头"""
        return nn.Sequential(
            nn.Linear(self.dino_model.embed_dim, self.num_features * 2),
            nn.BatchNorm1d(self.num_features * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features * 2, self.num_features),
            nn.BatchNorm1d(self.num_features)
        )

    def forward(self, x):
        # 类型转换
        if x.dtype != next(self.parameters()).dtype:
            x = x.type_as(next(self.parameters()))
        
        # 提取DINO特征
        x = self.dino_model(x)  # 获取CLS token
        
        # 特征头
        if self.feature_head is not None:
            x = self.feature_head(x)
        
        # 测试时直接返回归一化特征
        if not self.training:
            return F.normalize(x, p=2, dim=1)
        
        # 训练时返回特征和分类结果
        if self.classifier is not None:
            return F.normalize(x, p=2, dim=1), self.classifier(x)
        return F.normalize(x, p=2, dim=1)

    def float(self):
        super().float()
        self.dino_model.float()
        return self