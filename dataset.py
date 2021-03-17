import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class TrainDataset(Dataset):
    def __init__(self, content_dir):
        super(TrainDataset, self).__init__()
        self.content_dir = content_dir
        self.content_name_list = self.get_name_list(self.content_dir)
        self.transforms = self.transform()

    def get_name_list(self, name):
        name_list = os.listdir(name)
        name_list = [os.path.join(name, i) for i in name_list]
        np.random.shuffle(name_list)
        return name_list

    def transform(self):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess

    def __len__(self):
        a = len(self.content_name_list)
        return a

    def __getitem__(self, item):
        img = Image.open(self.content_name_list[item]).convert('RGB')
        img_out = self.transforms(img)
        return img_out