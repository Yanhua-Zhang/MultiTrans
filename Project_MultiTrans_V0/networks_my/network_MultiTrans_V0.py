import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet34
from torchvision.models import resnet50

from .backbone import resnet18_Deep, resnet50_Deep   # 加载预训练模型与参数
from .module import BasicLayer_Share_spatial_reduction, Conv2d_BN, local_global_Fusion_Average, SFNet_warp_grid, ConvModule

import math

class MultiTrans_V0(nn.Module):
    def __init__(self,  
                    config,
                    classes=2,                     
                    act_layer=nn.ReLU6,

                    ):
        super(MultiTrans_V0, self).__init__()

        self.use_dilation = config.use_dilation  
    
        self.norm_cfg = config.norm_cfg 
        # self.init_cfg = config.init_cfg
        self.branch_choose = config.branch_choose

        self.key_value_ratios = config.key_value_ratios   # 控制各个 branch 中 self-attention 中 key 和 value 的 spatial reduction
        self.If_direct_sum_fusion = config.If_direct_sum_fusion
        self.If_direct_upsampling = config.If_direct_upsampling
        self.is_dw = config.is_dw
        
        self.branch_in_channels = config.branch_in_channels

        self.Multi_branch_concat_fusion = config.Multi_branch_concat_fusion

        self.If_Local_GLobal_Fuison = config.If_Local_GLobal_Fuison

        self.If_Deep_Supervision = config.If_Deep_Supervision

        self.Dropout_Rate_CNN = config.Dropout_Rate_CNN
        self.Dropout_Rate_Trans = config.Dropout_Rate_Trans
        self.Dropout_Rate_SegHead = config.Dropout_Rate_SegHead # seg head 中的 Dropout rate 可以设低一点。

        # -------------------------------------------------------------------
        # backbone 加载
        if config.backbone_name == 'resnet18_Deep':
            resnet = resnet18_Deep(config.If_pretrained)
            stage_channels = [32, 64, 128, 256, 512]  # 'resnet18' 各 stage 的输出 channel

        elif config.backbone_name == 'resnet50_Deep':
            resnet = resnet50_Deep(config.If_pretrained)
            stage_channels = [128, 256, 512, 1024, 2048]  # 'resnet18' 各 stage 的输出 channel
        
        elif config.backbone_name == 'resnet50':
            resnet = resnet50()
            if config.If_pretrained:
                resnet.load_state_dict(torch.load('/home/zhangyanhua/Code_python/model_pretrained/resnet50-19c8e357.pth'))
            stage_channels = [128, 256, 512, 1024, 2048]  # 各 stage 的输出 channel

        # 利用 imagenet 预训练层构建 backbone
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool  
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4 
        del resnet  # 这里删除变量名，释放内存


        self.drop = nn.Dropout2d(self.Dropout_Rate_CNN)  # 这里仿照 TransFuse 尝试给每个 CNN stage 后面加入一个 Dropout2d
        
        # -------------------------------------------------------------------
        # Backbone 中 layer3,layer4 的 4 个 conv 层替换为空洞卷积
        if self.use_dilation:
        # if False:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)

        # --------------------------------------------------------------------
        # Backbone 各 stage 的 feature 统一进行 channel 的调整
        self.Backbone_channel_changes = []   # 顺序： [stage1_feature, stage2_feature, stage3_feature, stage4_feature]
        # for stage_channel in stage_channels:
        # for i in range(len(stage_channels)):
        for i in self.branch_choose:
            self.Backbone_channel_changes.append(nn.Sequential(
                nn.Conv2d(stage_channels[i], config.branch_in_channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(config.branch_in_channels[i]),
                nn.ReLU(inplace=True),
                ))

        self.Backbone_channel_changes = nn.ModuleList(self.Backbone_channel_changes) 

        # ==================================================================================
        # 构建多个 Trans 层
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depths)]  # stochastic depth decay rule
        # 构建 4 个 decoder branch
        # stage_key_channels = [8, 16, 32, 64]
        self.trans = nn.ModuleList()
        for i in self.branch_choose:
            self.trans.append(BasicLayer_Share_spatial_reduction(
                block_num=config.depths,                     # MSA 的层数
                embedding_dim=config.branch_in_channels[i],      # 输入 Transformer 的 channel 个数。
                key_dim=config.branch_key_channels[i],  
                num_heads=config.num_heads,
                Spatial_ratio=config.Spatial_ratios[i],       # 控制各个 branch 中整个 self-attention 中输入 feature 的 spatial reduction
                key_value_ratio = config.key_value_ratios[i], # 控制各个 branch 中 self-attention 中 key 和 value 的 spatial reduction
                mlp_ratio=config.mlp_ratios,
                attn_ratio=config.attn_ratios,

                drop=config.Dropout_Rate_Trans,      # Trans 中使用 dropout 
                drop_path=dpr,   # Trans 中使用 drop path：stochastic depth decay

                one_kv_head = config.one_kv_head, 
                share_kv = config.share_kv,
                If_efficient_attention = config.If_efficient_attention,
                norm_cfg=config.norm_cfg,
                act_layer=act_layer))
        
        # ==================================================================================
        # 选择 local, global feature fusion 的方式
        if self.If_Local_GLobal_Fuison:
            if self.If_direct_sum_fusion:
                # ------------------------------------------
                # 对 local 和 global feature 进行 sum 后进行 channel 调整（相当于一个 line fusion）。
                self.channels_change = nn.ModuleList()
                for i in self.branch_choose:
                    self.channels_change.append(Conv2d_BN(config.branch_in_channels[i], config.branch_out_channels, 1, norm_cfg=config.norm_cfg))
            else:
                # ------------------------------------------
                self.local_global_Fusions = nn.ModuleList()
                for i in self.branch_choose:
                    # 2 种融合方式可选：TopFormer 和 Average
                    self.local_global_Fusions.append(local_global_Fusion_Average(config.branch_in_channels[i], config.branch_out_channels))
        else:
            # ------------------------------------------
            # 对 global feature 直接进行 channel 调整（相当于一个 line fusion）。
            self.channels_change = nn.ModuleList()
            for i in self.branch_choose:
                self.channels_change.append(Conv2d_BN(config.branch_in_channels[i], config.branch_out_channels, 1, norm_cfg=config.norm_cfg))

        # ===================================================================================
        if not self.If_direct_upsampling:
            # ------------------------------------------
            # 计算不同 branch 中输出的 feature 间的 offset map 然后用于上采样
            self.stages_offset = nn.ModuleList()
            # stage 1 与 stage 2、stage 2与 stage 3、stage 3 与 stage 4 间的 offset maps
            for i in range(len(self.branch_choose)-1):   
                    self.stages_offset.append(SFNet_warp_grid(config.branch_in_channels[i], config.branch_in_channels[i]//2))

        # ===================================================================================
        # 各 branch 的 features sum 后进行 line fusion
        # self.linear_fuse = ConvModule(
        #     a = config.branch_out_channels,
        #     b = config.branch_out_channels,
        #     ks = 1,
        #     stride = 1,
        #     pad=0, 
        #     dilation=1,
        #     groups = config.branch_out_channels if self.is_dw else 1,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=dict(type='ReLU')
        # )
        if self.Multi_branch_concat_fusion:
            self.linear_fuse = nn.Sequential(
                nn.Conv2d(config.branch_out_channels*len(self.branch_choose), config.branch_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = config.branch_out_channels if self.is_dw else 1, bias=False),
                nn.BatchNorm2d(config.branch_out_channels),
                nn.ReLU(),
                )
        else:
            self.linear_fuse = nn.Sequential(
                nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = config.branch_out_channels if self.is_dw else 1, bias=False),
                nn.BatchNorm2d(config.branch_out_channels),
                nn.ReLU(),
                )

        # ------------------------------------------------------------------------
        # MMSegmenation 中的 seg head
        if config.If_Deep_Supervision:
            self.seg_head = nn.Sequential(
                        nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(config.branch_out_channels),
                        nn.ReLU(),
                        nn.Dropout2d(self.Dropout_Rate_SegHead),
                        nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                        )
            if self.training:
                self.branches_head = nn.ModuleList()
                for i in self.branch_choose:
                    self.branches_head.append(nn.Sequential(
                        nn.Conv2d(config.branch_out_channels, config.branch_out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(config.branch_out_channels),
                        nn.ReLU(),
                        nn.Dropout2d(self.Dropout_Rate_SegHead),
                        nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                        ))

        else:
            self.seg_head = nn.Sequential(
                        # nn.Conv2d(fam_dim, fam_dim, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(fam_dim),
                        # nn.ReLU(),
                        nn.Dropout2d(self.Dropout_Rate_SegHead),   # 这个也相关与在 multi-branch fusion 后的 Dropout     
                        nn.Conv2d(config.branch_out_channels, classes, kernel_size=1, stride=1, padding=0, bias=True)
                        )
            
    # ------------------------------------------------------------------------
    # 进行初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 一种卷积核初值化方法
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        # # 断点加载
        # if isinstance(self.pretrained, str):
        #     logger = get_root_logger()
        #     checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
        #     if 'state_dict_ema' in checkpoint:
        #         state_dict = checkpoint['state_dict_ema']
        #     elif 'state_dict' in checkpoint:
        #         state_dict = checkpoint['state_dict']
        #     elif 'model' in checkpoint:
        #         state_dict = checkpoint['model']
        #     else:
        #         state_dict = checkpoint
        #     self.load_state_dict(state_dict, False)


    def forward(self, x, y=None):
        x_size = x.size()

        # 把 single channel 的 slice 扩展为 3 通道的
        # 有的数据集是 grey image
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # 变成 3 Channel 的输入。在这里啊！！！把 single channel 的 slice 扩展为 3 通道的。
        
        x = self.layer0(x)
        stage0_feature = x

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.drop(x)
        stage1_feature = x

        x = self.layer2(x)
        x = self.drop(x)
        stage2_feature = x

        x = self.layer3(x)  # 用于计算中间层 loss
        x = self.drop(x)
        stage3_feature = x

        x = self.layer4(x)
        x = self.drop(x)
        stage4_feature = x
      
        # --------------------------------------------------------------
        stage_ouputs = [stage0_feature, stage1_feature, stage2_feature, stage3_feature, stage4_feature]  # 各特征维度 [32, 64, 128, 160]

        # --------------------------------------------------------------
        # 进行 backbone 各 stage channel 的 dim 调整
        compress_stage_features = []
        # for i in range(len(stage_ouputs)):
        j = 0
        for i in self.branch_choose:
            compress_stage_features.append(self.Backbone_channel_changes[j](stage_ouputs[i]))
            j = j + 1
        
        # ----------------------------------------------------------------------------------
        # 进行各个 branch 中 local 和 global 的 feature fusion
        branch_outs = []
        j = 0
        for i in self.branch_choose:
            global_out = self.trans[j](compress_stage_features[j])      # 4 个 Trans branch 

            if self.If_Local_GLobal_Fuison:
                if self.If_direct_sum_fusion:
                    fuse = global_out + compress_stage_features[j]
                    out_fuse = self.channels_change[j](fuse)  # 输出的 channel 进行调整
                    # out_fuse = fuse
                else:
                    out_fuse = self.local_global_Fusions[j](compress_stage_features[j], global_out)
                    # local_feat = compress_stage_features[i]
                    # global_feat = global_out
                    # global_weight = nn.functional.adaptive_avg_pool2d(global_feat, (1,1))
                    # out_fuse = local_feat * global_weight + global_feat
            else:
                out_fuse = self.channels_change[j](global_out)  # 对 Global feature 的 channel 进行调整
            
            branch_outs.append(out_fuse)
            j = j+1

        # ----------------------------------------------------------------------------------
        # 加入 deep supervison
        if self.training:
            if self.If_Deep_Supervision:
                deep_supervison_outs = []
                j = 0
                for i in self.branch_choose:
                    branch_pre = self.branches_head[j](branch_outs[j])    # 得到各个 branch 的 logist
                    branch_pre = F.interpolate(branch_pre, x_size[2:], mode='bilinear', align_corners=True)  # 上采样
                    deep_supervison_outs.append(branch_pre)
                    
                    j = j+1

        # ----------------------------------------------------------------------------------
        # 进行不同 branch feature 间的 spsample        
        if not self.If_direct_upsampling:
            # --------------------
            # 计算 stage 1 与 stage 2、stage 2 与 stage 3、stage 3 与 stage 4 间的 offset maps
            stages_warp_grid = []
            for i in range(len(branch_outs)-1):
                stages_warp_grid.append(self.stages_offset[i](branch_outs[i], branch_outs[i+1]))

            # 利用 offset maps 对各 branch 的 fuse_features 进行上采样。渐进式上采样。
            for i in range(1, len(branch_outs)):
                for k in reversed(range(i)):
                    branch_outs[i] = F.grid_sample(branch_outs[i], stages_warp_grid[k], align_corners=True)  # 利用网格法进行 scor map 的 upsample
        else:
            # ---------------------
            # 直接上采样
            for i in range(1, len(branch_outs)): 
                branch_outs[i] = F.interpolate(branch_outs[i], (branch_outs[0].size())[2:], mode='bilinear', align_corners=True)  # align_corners=False

        # ----------------------------------------------------------------------------------
        # 各 branch feature 的 fusion
        if self.Multi_branch_concat_fusion:
            sum_feature = torch.cat(branch_outs, 1)   # 进行 channel 维度的 concat fusion
            final_feature = self.linear_fuse(sum_feature)            
        else:
            sum_feature = sum(branch_outs)    
            final_feature = self.linear_fuse(sum_feature)

        # -----------------------------------------------------------------------------------
        # 计算 prediction map
        logits = self.seg_head(final_feature)

        logits = F.interpolate(logits, x_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            if self.If_Deep_Supervision:

                return logits, deep_supervison_outs
            else:
                return logits       
        else:
            return logits 

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    