import ml_collections

def get_config_UTNet():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    
    config.backbone_name='resnet50'
   
    config.version = 'without_Pretrain'
    

    return config



