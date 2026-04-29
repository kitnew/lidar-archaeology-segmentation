import segmentation_models_pytorch as smp

def create_model(
    backbone: str = 'resnet101',
    p_weights: bool = False,
    pretrained: bool = False,
    in_channels: int = 3,
    num_classes: int = 1,
    eval: bool = False
):
    """
    Create DeepLabV3+ model using segmentation-models-pytorch.
    
    Args:
        backbone (str): Encoder version. Can be any encoder from SMP (e.g., 'resnet101', 'resnet50').
        p_weights (bool): Alias for pretrained.
        pretrained (bool): If True, uses pre-trained weights for the encoder.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        eval (bool): If True, sets the model to evaluation mode.
    
    Returns:
        torch.nn.Module: The created DeepLabV3+ model.
    """
    
    encoder_weights = 'imagenet' if (pretrained or p_weights) else None
    
    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    
    if eval:
        model.eval()
    else:
        model.train()
    
    return model
