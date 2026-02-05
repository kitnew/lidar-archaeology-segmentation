import torch
def create_model(backbone: str = 'resnet101', p_weights = False, pretrained: bool = False, eval: bool = False):
    """
    Create DeepLabV3 model.
    
    Args:
        backbone (str): ResNet version. Can be 'resnet101' or 'resnet50'.
        pretrained (bool): If True, uses pre-trained weights.
        eval (bool): If True, sets the model to evaluation mode.
    
    Returns:
        torch.nn.Module: The created DeepLabV3 model.
    """
    
    if backbone == 'resnet101':
        from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
        from torchvision.models import ResNet101_Weights
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained or p_weights else None
        model = deeplabv3_resnet101(weights_backbone=weights, num_classes=1)
    elif backbone == 'resnet50':
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained or p_weights else None
        model = deeplabv3_resnet50(weights_backbone=weights, num_classes=1)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    if eval:
        model.eval()
    else:
        model.train()
    
    return model