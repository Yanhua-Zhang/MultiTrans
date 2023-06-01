import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_Polyp import Polyp_dataset

from tester import inference_Synapse, inference_Polyp


parser = argparse.ArgumentParser()

# ---------------------------------------------------------------------------
# 模型名字、数据集名字
parser.add_argument('--Model_Name', type=str, default='My_MultiTrans_V0', help='experiment_name')

parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')

# ----------------------------------------------------------------------------
parser.add_argument('--backbone', type=str, default='resnet50_Deep', help='experiment_name')
parser.add_argument('--use_dilation',type=str, default='False', help='use_dilation')
parser.add_argument('--If_pretrained',type=str, default='True', help='If_pretrained')

parser.add_argument('--branch_key_channels', nargs='+', type=int, help='branch_key_channels')
parser.add_argument('--branch_in_channels', nargs='+', type=int, help='branch_in_channels')
parser.add_argument('--branch_out_channels', type=int, help='branch_out_channels')
parser.add_argument('--branch_choose', nargs='+', type=int, help='branch_choose')


# parser.add_argument('--use_dilation',type=bool, default=False, help='use_dilation')  # 注意 bool 型的参数传入没有意义
# parser.add_argument('--If_direct_sum_fusion',type=bool, default=True, help='If_direct_sum_fusion')
  
parser.add_argument('--one_kv_head',type=str, default='True', help='one_kv_head')
parser.add_argument('--share_kv',type=str, default='True', help='share_kv')
parser.add_argument('--If_efficient_attention',type=str, default='True', help='If_efficient_attention')

parser.add_argument('--Multi_branch_concat_fusion',type=str, default='False', help='Multi_branch_concat_fusion')
parser.add_argument('--If_direct_sum_fusion',type=str, default='True', help='If_direct_sum_fusion')
parser.add_argument('--If_Local_GLobal_Fuison',type=str, default='True', help='If_Local_GLobal_Fuison')

parser.add_argument('--If_Deep_Supervision',type=str, default='False', help='If_Deep_Supervision')
parser.add_argument('--bran_weights', nargs='+', type=float, help='bran_weights')

# parser.add_argument('--weight_bran1', type=float, default=0.4, help='weight_bran1')
# parser.add_argument('--weight_bran2', type=float, default=0.3, help='weight_bran2')
# parser.add_argument('--weight_bran3', type=float, default=0.2, help='weight_bran3')
# parser.add_argument('--weight_bran4', type=float, default=0.1, help='weight_bran4')

parser.add_argument('--Dropout_Rate_CNN', type=float, default=0.2, help='Dropout_Rate_CNN')
parser.add_argument('--Dropout_Rate_Trans', type=float, default=0, help='Dropout_Rate_Trans')
parser.add_argument('--Drop_Path_Rate', type=float, default=0.1, help='Drop_Path_Rate')
parser.add_argument('--Dropout_Rate_SegHead', type=float, default=0.1, help='Dropout_Rate_SegHead')


# ---------------------------------------------------------------------------
# 专门为 TransFuse 加入的控制参数：
parser.add_argument('--Scale_Choose', type=str, default='Scale_L', help='Scale_Choose')

# ---------------------------------------------------------------------------
# 换用不同的 optimizer：
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
parser.add_argument('--grad_clip', type=float, default=0.5, help='gradient clipping norm')
parser.add_argument('--loss_name', type=str, default='ce_dice_loss', help='loss function')

parser.add_argument('--If_binary_prediction',type=str, default='False', help='If_binary_prediction')

parser.add_argument('--If_Multiscale_Train',type=str, default='True', help='If_Multiscale_Train')

# ---------------------------------------------------------------------------
# 模型训练的超参数
# 352
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')

parser.add_argument('--img_size_width', type=int, default=224, help='input patch size of network input')

parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')

parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')   

parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')

# -----------------
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')    # 这个 deterministic training 是啥原理？ 

parser.add_argument('--seed', type=int, default=1294, help='random seed')
# ---------------------------------------------------------------------------
# 是否保存可视化 results
# parser.add_argument('--is_savenii',type=bool, default=True, action="store_true", help='whether to save results during inference')
parser.add_argument('--is_savenii',type=bool, default=False, help='whether to save results during inference')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')

args = parser.parse_args()

# -----------------------------------------------------------
if args.use_dilation == 'False':
    args.use_dilation = False
else:
    args.use_dilation = True

if args.If_direct_sum_fusion == 'False':
    args.If_direct_sum_fusion = False
else:
    args.If_direct_sum_fusion = True

if args.one_kv_head == 'False':
    args.one_kv_head = False
else:
    args.one_kv_head = True

if args.share_kv == 'False':
    args.share_kv = False
else:
    args.share_kv = True

if args.If_efficient_attention == 'False':
    args.If_efficient_attention = False
else:
    args.If_efficient_attention = True

if args.Multi_branch_concat_fusion == 'False':
    args.Multi_branch_concat_fusion = False
else:
    args.Multi_branch_concat_fusion = True

if args.If_Local_GLobal_Fuison == 'False':
    args.If_Local_GLobal_Fuison = False
else:
    args.If_Local_GLobal_Fuison = True

if args.If_binary_prediction == 'False':
    args.If_binary_prediction = False
else:
    args.If_binary_prediction = True

if args.If_Multiscale_Train == 'False':
    args.If_Multiscale_Train = False
else:
    args.If_Multiscale_Train = True

if args.If_Deep_Supervision == 'False':
    args.If_Deep_Supervision = False
else:
    args.If_Deep_Supervision = True

if args.If_pretrained == 'False':
    args.If_pretrained = False
else:
    args.If_pretrained = True

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../preprocessed_data/Synapse/test_vol_h5', # 换绝对路径
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,  # 这个是啥？？
        },
        'Polyp': {
            'Dataset': Polyp_dataset,    # 用于加载 Polyp 数据集的函数/类函数
            'volume_path': '/home/zhangyanhua/Code_python/Dataset/Medical_Dataset/Polyp/TestDataset',  # 这个是 test 数据集的地址
            'list_dir': './lists/lists_Polyp',
            'num_classes': 2,
            'z_spacing': 1,  # 这个是啥？？
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    if args.If_binary_prediction:  # 如果采用 2 值计算的话
        args.num_classes = 1

    # ================================================================================
    # 实例化网络模型
    if args.Model_Name == 'TransUNet':

        from networks_TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks_TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        # ---------------------------------------------------------------
        args.vit_name = 'R50-ViT-B_16'

        config_vit = CONFIGS_ViT_seg[args.vit_name]   # 获取网络结构的具体参数
        config_vit.n_classes = args.num_classes
        
        if args.vit_name.find('R50') !=-1:
            config_vit.patches.grid = (int(args.img_size/config_vit.vit_patches_size), int(args.img_size/config_vit.vit_patches_size))    

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'TU_' + dataset_name + str(args.img_size)
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/TU_pretrain' if args.is_pretrain else '/TU'
        snapshot_path += '_' + args.vit_name
        snapshot_path = snapshot_path + '_skip' + str(config_vit.n_skip)  # 用了几个 skip connect

        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path + '_vitpatch' + str(config_vit.vit_patches_size) if config_vit.vit_patches_size!=16 else snapshot_path
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
        Model_path = Model_path + snapshot_path 


        # 实例化网络，并加载到 GPU 上    
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

        # 加载训练好的模型，从 best_model 或 epoch_(max_epochs-1) 中加载
        snapshot = os.path.join(Model_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = Model_path.split('/')[-1]

    elif args.Model_Name == 'My_MultiTrans_V0':
        from networks_my.configs_MultiTrans_V0 import get_config_MultiTrans_V0
        from networks_my.network_MultiTrans_V0 import MultiTrans_V0

        config = get_config_MultiTrans_V0()

        config.backbone_name = args.backbone   # 替换为交互窗口的输入的 backbone
        if args.branch_key_channels is not None:
            config.branch_key_channels = args.branch_key_channels
        config.use_dilation = args.use_dilation
        config.If_direct_sum_fusion = args.If_direct_sum_fusion

        if args.branch_in_channels is not None:
            config.branch_in_channels = args.branch_in_channels
            
        if args.branch_out_channels is not None:
            config.branch_out_channels = args.branch_out_channels

        if args.branch_choose is not None:
            config.branch_choose = args.branch_choose

        if args.one_kv_head is not None:
            config.one_kv_head = args.one_kv_head

        if args.share_kv is not None:
            config.share_kv = args.share_kv

        if args.If_efficient_attention is not None:
            config.If_efficient_attention = args.If_efficient_attention

        if args.Multi_branch_concat_fusion is not None:
            config.Multi_branch_concat_fusion = args.Multi_branch_concat_fusion

        if args.If_Local_GLobal_Fuison is not None:
            config.If_Local_GLobal_Fuison = args.If_Local_GLobal_Fuison

        config.If_Deep_Supervision = args.If_Deep_Supervision

        config.If_pretrained = args.If_pretrained

        config.Dropout_Rate_CNN = args.Dropout_Rate_CNN
        config.Dropout_Rate_Trans = args.Dropout_Rate_Trans
        config.Dropout_Rate_SegHead = args.Dropout_Rate_SegHead

        config.drop_path_rate = args.Drop_Path_Rate

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'My_MultiTrans_V0_' + dataset_name + str(args.img_size)
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/My_MultiTrans_V0_pretrain' if args.is_pretrain else '/My_MultiTrans_V0'
        snapshot_path += '_' + config.backbone_name
        snapshot_path = snapshot_path + '_' + config.version
       
        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path 
        Model_path = Model_path + snapshot_path 

        # ---------------------------------------------------------------
        # 实例化网络
        net = MultiTrans_V0(config, classes=args.num_classes).cuda()  # 网络实例化

        if args.n_gpu > 1:
            # print('使用多 GPU')
            # model = nn.DataParallel(model, device_ids=[0,1])
            net = nn.DataParallel(net)

        # 加载训练好的模型，从 best_model 或 epoch_(max_epochs-1) 中加载
        snapshot = os.path.join(Model_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = Model_path.split('/')[-1]

    elif args.Model_Name == 'UTNet':
        from networks_UTNet.utnet import UTNet, UTNet_Encoderonly   # 加载自己的模型
        from networks_UTNet.configs_UTNet import get_config_UTNet

        config = get_config_UTNet()

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'UTNet_' + dataset_name + str(args.img_size)

        Log_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Log')
        TensorboardX_path = "../Results/model_Trained/{}/{}".format(args.exp, 'TensorboardX')
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')
        Summary_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Summary')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/UTNet_pretrain' if args.is_pretrain else '/UTNet'
        snapshot_path += '_' + config.backbone_name
        snapshot_path = snapshot_path + '_' + config.version
       
        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path  

        # --------------------------------
        Model_path = Model_path + snapshot_path 

        # --------------------------------
        net = UTNet(1, base_chan=32, num_classes=args.num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True).cuda()

        # 加载训练好的模型，从 best_model 或 epoch_(max_epochs-1) 中加载
        snapshot = os.path.join(Model_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = Model_path.split('/')[-1]

    elif args.Model_Name == 'TransFuse':
        from networks_TransFuse.TransFuse import TransFuse_S, TransFuse_L, TransFuse_L_384

        # ---- build models ----
        if args.Scale_Choose == 'Scale_S':
            backbone_name = 'resnet34'
            version = 'Scale_S'
        elif args.Scale_Choose == 'Scale_L':
            backbone_name = 'resnet50'
            version = 'Scale_L'
        elif args.Scale_Choose == 'Scale_L_384':
            backbone_name = 'resnet50'
            version = 'Scale_L_384'

        # ---------------------------------------------------------------
        # 生成 保存数据集结果 的文件名
        args.exp = 'TransFuse_' + dataset_name + str(args.img_size)

        Log_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Log')
        TensorboardX_path = "../Results/model_Trained/{}/{}".format(args.exp, 'TensorboardX')
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')
        Summary_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Summary')

        # --------------------------------
        # 创建文件名
        snapshot_path = '/TransFuse_pretrain' if args.is_pretrain else '/TransFuse'
        snapshot_path += '_' + backbone_name
        snapshot_path = snapshot_path + '_' + version
       
        # 如果不是默认值，就把修改值在文件名中体现出来
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)

        # 可以通过改 seed 来进行重复实验
        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path  

        # --------------------------------
        Model_path = Model_path + snapshot_path 

        # ---- build models ----
        if args.Scale_Choose == 'Scale_S':
            net = TransFuse_S(num_classes=args.num_classes, pretrained=True).cuda()
        elif args.Scale_Choose == 'Scale_L':
            net = TransFuse_L(num_classes=args.num_classes, pretrained=True).cuda()
        elif args.Scale_Choose == 'Scale_L_384':
            net = TransFuse_L_384(num_classes=args.num_classes, pretrained=True).cuda()

        # 加载训练好的模型，从 best_model 或 epoch_(max_epochs-1) 中加载
        snapshot = os.path.join(Model_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = Model_path.split('/')[-1]


    log_folder = '../Results/test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # 是否保存可视化结果图片
    if args.is_savenii:
        args.test_save_dir = '../Results/predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inferencer = {'Synapse': inference_Synapse, 'Polyp': inference_Polyp}
    # inferencer[dataset_name](args, net, test_save_path)

    if dataset_name == 'Synapse':
        inference_Synapse(args, net, test_save_path)
    elif dataset_name == 'Polyp':
        inference_Polyp(args, net, split='CVC-ClinicDB_test', test_save_path=test_save_path)
        inference_Polyp(args, net, split='Kvasir_test', test_save_path=test_save_path)
        inference_Polyp(args, net, split='CVC-ColonDB_test', test_save_path=test_save_path)
        inference_Polyp(args, net, split='ETIS-LaribPolypDB_test', test_save_path=test_save_path)
        inference_Polyp(args, net, split='CVC-300_test', test_save_path=test_save_path)
