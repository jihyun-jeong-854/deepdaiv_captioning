from PIL import Image
from torchvision import transforms

import torch
import argparse
import matplotlib.pyplot as plt
# from captioning_dataset import ImageFolderDataset, ImageListDatasetFromGradio
from torch.utils.data import Dataset, DataLoader
from image2text import vd_inference


def img2cap(*args, model, arguments):

    vd_infer = model

    img_list = []
    for i, img in enumerate(args):
        if (img != None) and (img != False) and (type(img) != bool):
            img_list.append(img)

#     cap_list = [vd_infer.inference_i2t(im, 20) for im in img_list]
    
    cap_list = []
    for im in img_list : 
        cap_ = vd_infer.inference_i2t(im,20)
        cap_list.extend(cap_)

    return cap_list
