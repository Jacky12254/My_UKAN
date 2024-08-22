#这个文件需要将四张同名的图片当成4通道处理，所以这里的channel是4
import os
import numpy as np
import tifffile as tiff
from PIL import Image
from skimage import io, transform
from skimage.util import img_as_ubyte
from tqdm import tqdm

# 定义源目录和目标目录
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建绝对路径
src_dir_opt_clear = os.path.join(current_dir, './origin_dataset/opt_clear')
src_dir_opt_cloudy = os.path.join(current_dir, './origin_dataset/opt_cloudy')
src_dir_sar_VH = os.path.join(current_dir, './origin_dataset/SAR/VH')
src_dir_sar_VV = os.path.join(current_dir, './origin_dataset/SAR/VV')
dst_dir = os.path.join(current_dir, 'img_pro_dataset')


if not os.path.exists(dst_dir):#如果目标文件夹不存在
    os.makedirs(dst_dir, exist_ok=True)#创建文件夹

# 获取源目录中所有图像文件的列表
image_files_opt_clear = [f for f in os.listdir(src_dir_opt_clear) if os.path.isfile(os.path.join(src_dir_opt_clear, f))]#image_flies是一个列表，里面存储了所有的图片文件名
image_files_opt_cloudy = [f for f in os.listdir(src_dir_opt_cloudy) if os.path.isfile(os.path.join(src_dir_opt_cloudy, f))]
image_files_sar_VH = [f for f in os.listdir(src_dir_sar_VH) if os.path.isfile(os.path.join(src_dir_sar_VH, f))]
image_files_sar_VV = [f for f in os.listdir(src_dir_sar_VV) if os.path.isfile(os.path.join(src_dir_sar_VV, f))]

for image_file_index in tqdm(range(len(image_files_opt_clear)), desc='Processing images'):#tqdm是一个进度条库，这里是显示进度条
    # Load the image
    image_file_opt_clear = image_files_opt_clear[image_file_index]#image_file是一个字符串，存储了图片文件名
    image_file_opt_cloudy = image_files_opt_cloudy[image_file_index]
    image_file_sar_VH = image_files_sar_VH[image_file_index]
    image_file_sar_VV = image_files_sar_VV[image_file_index]

    image_file = image_file_opt_clear.replace('.png', '.tif')
    #这里是将opt_clear的图片作为基准，所以这里的image_file是opt_clear的图片文件名

    image_opt_clear = io.imread(os.path.join(src_dir_opt_clear, image_file_opt_clear))
    image_opt_cloudy = io.imread(os.path.join(src_dir_opt_cloudy, image_file_opt_cloudy))
    image_sar_VH = io.imread(os.path.join(src_dir_sar_VH, image_file_sar_VH))

    image_sar_VH = np.expand_dims(image_sar_VH, axis=-1)
    image_sar_VV = io.imread(os.path.join(src_dir_sar_VV, image_file_sar_VV))
    image_sar_VV = np.expand_dims(image_sar_VV, axis=-1)

    assert image_opt_clear.shape[:2] == image_opt_cloudy.shape[:2] == image_sar_VH.shape[:2] == image_sar_VV.shape[:2], "所有图片的尺寸必须一致"
    multi_channel_image = np.concatenate((image_opt_clear, image_opt_cloudy, image_sar_VH, image_sar_VV), axis=-1)
    tiff.imwrite(os.path.join(dst_dir, image_file), multi_channel_image)#path是保存的路径，multi_channel_image是要保存的图片,path.join是将多个路径组合后返回，image_file是图片文件名