from tool.train import train, eval
import os
import argparse
import torch
import numpy as np
from tool.img2img import img2img
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main(model_config = None, state = "train"):
    if state == "train":
        if model_config is not None:
            modelConfig = model_config
        if modelConfig["state"] == "train":
            train(modelConfig)
            modelConfig['batch_size'] = 8
            modelConfig['test_load_weight'] = 'ckpt_{}_.pt'.format(modelConfig['epoch'])
            for i in range(32):
                modelConfig["sampledImgName"] = "sampledImgName{}.png".format(i)
                eval(modelConfig)
        else:
            for i in range(32):
                modelConfig["sampledImgName"] = "sampledImgName{}.png".format(i)
                eval(modelConfig)
    else:
        img2img(modelConfig)

def seed_all(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default='train') # train or eval
    parser.add_argument('--dataset', type=str, default='ROIs') # busi, glas, cvc
    parser.add_argument('--epoch', type=int, default=1000) # 1000 for cvc/glas, 5000 for busi
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--channel', type=int, default=64) # 64 or 128，这里具体是指通道数，也就是特征图的数量
    parser.add_argument('--test_load_weight', type=str, default='ckpt_1000_.pt')
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=float, default=64) #default 64是指图片的大小，default的意思是默认值
    parser.add_argument('--dataset_repeat', type=int, default=1) # did not use
    parser.add_argument('--seed', type=int, default=0) # did not use
    parser.add_argument('--model', type=str, default='UKAN_Hybrid')
    parser.add_argument('--exp_nme', type=str, default='UKAN_Hybrid')
    parser.add_argument('--save_root', type=str, default='./model/') 
    parser.add_argument('--output_root', type=str, default='./output/')
    parser.add_argument('--sampledImgName', type=str, default='ROIs_02_p.png')
    args = parser.parse_args()#这个是解析参数的意思

    save_root = args.save_root
    if args.seed != 0:
        seed_all(args)

    modelConfig = {
        "dataset": args.dataset, 
        "state": args.state, # or eval
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "T": args.T,#T是指迭代的次数
        "channel": args.channel,
        "channel_mult": [1, 2, 3, 4, 5, 6, 7, 8],#这个是通道数，这里是一个列表,这里是指通道数的倍数,有8个通道
        "attn": [2],
        "num_res_blocks": args.num_res_blocks,
        "dropout": args.dropout,
        "lr": args.lr,
        "multiplier": 2.,
        "beta_1": 1e-4,#这个是beta_1
        "beta_T": 0.02,
        "img_size": 256,#这个是图片的大小
        "grad_clip": 1.,
        "device": "cuda", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": os.path.join(save_root, args.exp_nme, "Weights"),#这个是保存模型的权重
        "sampled_dir": os.path.join(save_root, args.exp_nme, "Gens"),#这个是保存生成的图片
        "test_load_weight": args.test_load_weight,
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,#这个是指每行显示的图片数量
        "model":args.model,
        "version": 1,
        "dataset_repeat": args.dataset_repeat,
        "seed": args.seed,
        "save_root": args.save_root,
        "output_root": args.output_root,
        "sampledImgName": args.sampledImgName,
        }

    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)#创建文件夹
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)

    # backup 
    import shutil#shutil是一个文件操作的工具包
    shutil.copy("tool/Model_UKAN_Hybrid.py", os.path.join(save_root, args.exp_nme))#这里是为了保存模型的代码
    shutil.copy("tool/train.py", os.path.join(save_root, args.exp_nme))#这里是为了保存训练的代码

    main(modelConfig, 'train')#这里是调用main函数，开始训绿
