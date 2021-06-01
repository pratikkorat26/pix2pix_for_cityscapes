from torch.utils.data import Dataset , DataLoader
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

class Pix2PixDataset(Dataset):
    def __init__(self , image_dir , transform):
        self.image_dir = image_dir
        self.transform = transform

        if self.transform:
            self.transform_dict = {
                'real_image': A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=0.2),
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                    ToTensorV2(),
                ]),

                'seg_image': A.Compose([
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                    ToTensorV2()]),
            }

        self.image_list = glob(f"{self.image_dir}\\*.jpg")


    def __getitem__(self, item):
        img_file = self.image_list[item]
        image = np.array(Image.open(img_file))

        real_img = image[:, :256, :]
        assert real_img.shape == (256, 256, 3)
        seg_img = image[:, 256:, :]
        assert seg_img.shape == (256, 256, 3)

        real_img = self.transform_dict["real_image"](image=real_img)
        seg_img = self.transform_dict["seg_image"](image=seg_img)

        real_img = real_img["image"]
        seg_img = seg_img["image"]

        return real_img , seg_img

    def __len__(self):
        return  len(self.image_list)

if __name__ == '__main__':
    dataset = Pix2PixDataset(image_dir = "cityscapes" ,transform = True)
    dataloader = DataLoader(dataset , batch_size = 5 , shuffle = True)
    data = next(iter(dataloader))
    print(data[0].shape , data[1].shape)
