# dataset
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from config import both_transform, transform_only_input, transform_only_mask
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MapDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.list_files = os.listdir(path)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.path, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations['image']
        target_image = augmentations['image0']

        input_image = transform_only_input(image=input_image)['image']
        target_image = transform_only_mask(image=target_image)['image']

        return input_image, target_image


if __name__ == '__main__':
    dataset = MapDataSet('../input/pix2pix-dataset/maps/maps/train')
    loader = DataLoader(dataset, batch_size=3)
