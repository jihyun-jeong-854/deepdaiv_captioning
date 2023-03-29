import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

#


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_names = [file for file in os.listdir(
            folder_path) if not file.endswith('txt')]
#         self.image_names = os.listdir(folder_path)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_names)

# Gradio 형식에 맞게 수정


class ImageListDatasetFromGradio(Dataset):
    def __init__(self, image_list, transform=None):
        # image_list = [file for file in image_list if not file.endswith('txt')] # 결과 파일 제거
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, index):
        image = self.image_list[index]

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_list)
