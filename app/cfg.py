import argparse

def str2bool(value):
    """Parse explicit CLI booleans (argparse's ``type=bool`` is unsafe)."""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got: {value}")


def int_list(value):
    """Parse comma-separated integer lists, including an empty list."""
    if value.strip() == "":
        return []
    return [int(item.strip()) for item in value.split(",")]


def float_list(value):
    return [float(item.strip()) for item in value.split(",")]


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-arch', type=str, default='vit_b', help='net architecture, pick between vit_h, vit_b, vit_t')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-dataset_name', type=str, default='custom_dataset', help='the name of the dataset to fine-tune')
    
    parser.add_argument('-img_folder', type=str, default='/workspace/data', help='root folder containing images')
    parser.add_argument('-mask_folder', type=str, default='/workspace/data', help='root folder containing masks')
    parser.add_argument('-train_img_list', type=str, default='/workspace/data/train.csv')
    parser.add_argument('-val_img_list', type=str,default='/workspace/data/val.csv')
    parser.add_argument('-targets', type=str,default='combine_all')

    parser.add_argument('-finetune_type', type=str, default='adapter', help='normalization type, pick among vanilla,adapter,lora')
    parser.add_argument('-normalize_type', type=str, default='sam', help='normalization type, pick between sam or medsam')
    
    parser.add_argument('-dir_checkpoint', type=str, default='/workspace/checkpoints/run', help='checkpoint output folder')
    parser.add_argument('-num_cls', type=int, default=2, help='the number of output channels (need to be your target cls num +1)')
    parser.add_argument('-epochs', type=int, default=200, help='the number of largest epochs to train')
    parser.add_argument('-sam_ckpt', type=str, default='/workspace/weights/sam_vit_b_01ec64.pth', help='pretrained checkpoint path')
    
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=None, help='visualization')
    parser.add_argument('-reverse', type=str2bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=str2bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=100,help='interval between each validation')
    parser.add_argument('-gpu', type=str2bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=64, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-s', type=str2bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-if_warmup', type=str2bool, default=False, help='if warm up training phase')
    parser.add_argument('-warmup_period', type=int, default=200, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='isic' ,type=str,help='dataset name')
    parser.add_argument('-thd', type=str2bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')

    parser.add_argument('-if_update_encoder', type=str2bool, default=False , help='if update_image_encoder')
    parser.add_argument('-if_encoder_adapter', type=str2bool, default=False , help='if add adapter to encoder')
    
    parser.add_argument('-encoder-adapter-depths', '-encoder_adapter_depths', dest='encoder_adapter_depths', type=int_list, default=[0,1,10,11] , help='comma-separated encoder block indexes')
    parser.add_argument('-if_mask_decoder_adapter', type=str2bool, default=False , help='if add adapter to mask decoder')
    parser.add_argument('-decoder_adapt_depth', type=int, default=2, help='the depth of the decoder adapter')
    
    parser.add_argument('-if_encoder_lora_layer', type=str2bool, default=False , help='if add lora to encoder')
    parser.add_argument('-if_decoder_lora_layer', type=str2bool, default=False , help='if add lora to decoder')
    parser.add_argument('-encoder_lora_layer', type=int_list, default=[0,1,10,11] , help='comma-separated block indexes; empty means every layer')
    
    parser.add_argument('-if_split_encoder_gpus', type=str2bool, default=False , help='legacy model-parallel mode')
    parser.add_argument('-devices', type=int_list, default=[0,1] , help='comma-separated model-parallel GPU indexes')
    parser.add_argument('-gpu_fractions', type=float_list, default=[0.5,0.5] , help='comma-separated encoder split fractions')
    
  
    parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
    opt = parser.parse_args()

    return opt
