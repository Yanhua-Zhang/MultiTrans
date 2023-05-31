import ml_collections

def get_config_MultiTrans_V0():
    """Returns the MultiTrans_V0 configuration."""
    config = ml_collections.ConfigDict()

    # -------------------------------------------------------------------------------
    # branch channel, branch choose
    config.branch_choose = [1, 2, 3, 4]             # 将 multi-branch 改为可选。

    config.branch_key_channels = [8, 16, 32, 64, 128]   # [4, 8, 16, 32]。 4 个 branch 中 MSA 的 key 的 channel 数
    config.branch_in_channels = [128, 256, 512, 512, 1024]              # 从各个 branch 中输入的 channel 个数
    config.branch_out_channels = 256              # 进行各 branch feature fusion 后的 channel 个数

    # -------------------------------------------------------------------------------
    # self-attention module
    config.one_kv_head = True
    config.share_kv = True
    config.If_efficient_attention = True

    config.Spatial_ratios = [1, 1, 1, 1, 1]         # 控制各个 branch 中整个 self-attention 中输入 feature 的 spatial reductio
    config.key_value_ratios = [1, 1, 1, 1, 1]        # 控制各个 branch 中 self-attention 中 key 和 value 的 spatial reduction

    config.depths=4           # MSA 的层数
    config.num_heads=8        # MSA head 的个数
    config.attn_ratios=2      # key_dim 乘以 attn_ratios 是单个 value 的 channel 数
    config.mlp_ratios=2        # 用于计算 MLP 中隐含层的 channel 数

    # -------------------------------------------------------------------------------
    # drop out 
    config.drop_path_rate=0.1  # 是否在 Trans 中使用 drop_path
    config.Dropout_Rate_CNN = 0.2
    config.Dropout_Rate_Trans = 0
    config.Dropout_Rate_SegHead = 0.1

    # -------------------------------------------------------------------------------
    # backbone, network architecture
    config.backbone_name='resnet50_Deep'
    config.use_dilation=False

    config.If_direct_sum_fusion = True      # 选择 local global feature fusion 的方式
    config.If_direct_upsampling = True      # 选择 branch feature 间 upsample 的方式
    config.Multi_branch_concat_fusion = False

    # -------------------------------------------------------------------------------
    config.norm_cfg=dict(type='BN', requires_grad=True)
    config.is_dw = False

    config.version = 'V0'
    

    return config



