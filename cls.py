from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
import requests
#from generate import sequence_generator
import torch
import matplotlib.pyplot as plt
from captioning_dataset import ImageFolderDataset, ImageListDatasetFromGradio
from torch.utils.data import Dataset, DataLoader


def img2cls(*args, arguments):

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 224
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution),
                          interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Define dataset
    img_list = []
    for i, img in enumerate(args):
        if (img != None) and (img != False) and (type(img) != bool):
            img_list.append(img)
    dataset = ImageListDatasetFromGradio(
        img_list, transform=patch_resize_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define Model
    processor = ViTImageProcessor.from_pretrained(
        'google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224')
    cls_list = []
    for batch_idx, image in enumerate(dataloader):
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argsort(descending=True)[:, :3].squeeze()
        cls_result = [model.config.id2label[i] for i in predicted_class_idx]
        cls_list += cls_result
    print(cls_list)
