from __future__ import absolute_import
from .clip_model import CLIPReID  
from .vit_reid import DINOReID
from .resnet import *
from .resnet_rpp import resnet50_rpp

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
        'resnet50_rpp': resnet50_rpp,
        'clip_vit_b16': lambda **kwargs: CLIPReID(clip_model_name='ViT-B/16', **kwargs),
    'clip_vit_l14': lambda **kwargs: CLIPReID(clip_model_name='ViT-L/14', **kwargs),
    ##########################################
        'vit_small': lambda **kwargs: DINOReID(dino_model_name='vit_small', **kwargs),
    'vit_base': lambda **kwargs: DINOReID(dino_model_name='vit_base', **kwargs)
    ##########################################
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name.startswith('vit'):
        vit_kwargs = {k: v for k, v in kwargs.items() 
                     if k in ['num_features', 'num_classes']}
        return __factory[name](*args, **vit_kwargs)
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
