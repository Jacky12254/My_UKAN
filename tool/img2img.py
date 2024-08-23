import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from .Model_UKAN_Hybrid import UKan_Hybrid
from .Scheduler import GradualWarmupScheduler
from .Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from torchvision.utils import save_image
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    'UKan_Hybrid': UKan_Hybrid,
}#这个字典是用来存储模型的，key是模型的名字，value是模型的类




def img2img(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])

        current_dir = os.path.dirname('../dataset/test/')
        image_files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]

        model = model_dict[modelConfig["model"]](T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)

        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        
        Gaussion_std = 0.1
        Gaussion_mean = 0
        # Sampled from standard normal distribution, 
        # noisyImage = torch.randn(
        #     size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)     
        # Load the initial noise image from a .tif file
        for i in range(tqdm(range(len(image_files)), desc='generate images')):
            tif_image = Image.open('../dataset/test/ROIs_41_p_{}.tif'.format(i+3130))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),
            ])#这个是一个转换函数，将图片转换为tensor，然后resize到指定的大小
            noisyImage = transform(tif_image).unsqueeze(0).to(device)
            # Add batch dimension and move to device
            noise = torch.randn_like(noisyImage.size(), device=device) * Gaussion_std + Gaussion_mean
            noisyImage = noisyImage + noise
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
            sampledImgs = sampler(noisyImage)
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]，这里的0.5和0.5是为了将图片的像素值从[-1,1]转换到[0,1]

            for i, image in enumerate(sampledImgs):#enumerate是一个枚举函数，返回的是一个枚举对象
        
                save_image(image, os.path.join(modelConfig["output_root"],  modelConfig["sampledImgName"].replace('.png','_{}.png').format(i+3130)), nrow=modelConfig["nrow"])
                #图片最后保存的路径是modelConfig["sampled_dir"]，命名格式是modelConfig["sampledImgName"]_i.png