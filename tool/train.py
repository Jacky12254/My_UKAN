
import os
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, transforms
# from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from .Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
# from Diffusion.UNet import UNet, UNet_Baseline
# from Diffusion.Model_ConvKan import UNet_ConvKan
# from Diffusion.Model_UMLP import UMLP
from .Model_UKAN_Hybrid import UKan_Hybrid
from .Scheduler import GradualWarmupScheduler
from skimage import io
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset
import sys


model_dict = {
    # 'UNet': UNet,
    # 'UNet_ConvKan': UNet_ConvKan, # dose not work
    # 'UMLP': UMLP,
    'UKAN_Hybrid': UKan_Hybrid,
    # 'UNet_Baseline': UNet_Baseline,
}#这个字典是用来存储模型的，key是模型的名字，value是模型的类

class UnlabeledDataset(Dataset):#这是一个数据集类，用于加载数据
    def __init__(self, folder, transform=None, repeat_n=1):
        self.folder = folder
        self.transform = transform
        # self.image_files = os.listdir(folder) * repeat_n
        self.image_files = os.listdir(folder) 
        self.image_files = [file for file in tqdm(self.image_files, desc="Loading Dataset")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.folder, image_file)
        image = io.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.Tensor([0])


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    log_print = True
    if log_print:
        file = open(modelConfig["save_weight_dir"]+'log.txt', "w")#这里是打开一个文件，用于保存训练的过程，这里是将标准输出重定向到文件
        sys.stdout = file
    transform = Compose([
        ToTensor(),
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.RandomVerticalFlip(),#随机垂直翻转
        Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
        ])#归一化，参数为均值和标准差

    if modelConfig["dataset"] == 'ROIs':
        dataset = UnlabeledDataset('dataset/img_pro_dataset/', transform=transform, repeat_n=modelConfig["dataset_repeat"])
        #这里是加载数据集，这里是加载的是ROIs数据集,repeate_n是指重复的次数
    else:
        raise ValueError('dataset not found')

    print('modelConfig: ')
    for key, value in modelConfig.items():
        print(key, ' : ', value)
        
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=0, drop_last=True, pin_memory=True)#这里num_workers修改为0，因为在windows下num_workers不能大于0

    print('Using {}'.format(modelConfig["model"]))
    # model setup，这里是模型的初始化
    net_model =model_dict[modelConfig["model"]](T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))#这里是在训练的时候加载预训练的模型
        
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)#这里是在训练的时候使用的优化器，AdamW是Adam的一个变种，加入了权重衰减
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)#余弦退火学习率调度器
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)#学习率预热调度器

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for epoch in range(1,modelConfig["epoch"]+1):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()#梯度清零
                x_0 = images.to(device)#将数据放到GPU上
                
                loss = trainer(x_0).sum() / 1000.#计算loss
                loss.backward()#反向传播
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])#梯度裁剪，这里使用的是梯度的L2范数，clip_grad_norm_是对梯度进行裁剪，作用是防止梯度爆炸
                optimizer.step()#更新参数
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })#这里是在进度条上显示一些信息，epoch是当前的epoch，loss是当前的loss，img shape是当前的图片的shape，LR是当前的学习率
                # print version
                if log_print:
                    print("epoch: ", epoch, "loss: ", loss.item(), "img shape: ", x_0.shape, "LR: ", optimizer.state_dict()['param_groups'][0]["lr"])
                warmUpScheduler.step()#学习率调度器更新
                torch.cuda.empty_cache()
        if epoch % 50 ==0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(epoch) + "_.pt"))
            modelConfig['test_load_weight'] = 'ckpt_{}_.pt'.format(epoch)
            eval_tmp(modelConfig, epoch)

    torch.save(net_model.state_dict(), os.path.join(
        modelConfig["save_weight_dir"], 'ckpt_' + str(epoch) + "_.pt"))
    if log_print:
        file.close()
        sys.stdout = sys.__stdout__#这里是关闭文件，恢复标准输出
    
def eval_tmp(modelConfig: Dict, nme: int):#tmp是为了不覆盖原来的eval
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = model_dict[modelConfig["model"]](T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device, weights_only=True)
    
        model.load_state_dict(ckpt)
        
        #以下这一段是为了在测试的时候生成图片
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
            # modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        save_root = modelConfig["sampled_dir"].replace('Gens','Tmp')
        os.makedirs(save_root, exist_ok=True)
        save_image(sampledImgs, os.path.join(
            save_root,  modelConfig["sampledImgName"].replace('.png','_{}.png').format(nme)), nrow=modelConfig["nrow"])
        if nme < 0.95 * modelConfig["epoch"]:
            os.remove(os.path.join(
                modelConfig["save_weight_dir"], modelConfig["test_load_weight"]))
            
        torch.cuda.empty_cache()

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])

        model = model_dict[modelConfig["model"]](T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)

        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)     
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        for i, image in enumerate(sampledImgs):
    
            save_image(image, os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"].replace('.png','_{}.png').format(i)), nrow=modelConfig["nrow"])