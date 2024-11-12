import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.io import read_image, ImageReadMode

class Segmentation_Dataset(Dataset):
    def __init__(self, ImagesDirectory, MasksDirectory):
        self.ImagesDirectory = ImagesDirectory
        self.MasksDirectory = MasksDirectory
        self.image_paths = []
        self.mask_paths = []

        
        for city_folder in os.listdir(self.ImagesDirectory):
            city_image_path = os.path.join(self.ImagesDirectory, city_folder)
            city_mask_path = os.path.join(self.MasksDirectory, city_folder)

            if os.path.isdir(city_image_path) and os.path.isdir(city_mask_path):
                image_files = sorted(os.listdir(city_image_path))
                mask_files = sorted([f for f in os.listdir(city_mask_path) if 'labelIds' in f])

                for i in range(min(len(image_files), len(mask_files))):
                    self.image_paths.append(os.path.join(city_image_path, image_files[i]))
                    self.mask_paths.append(os.path.join(city_mask_path, mask_files[i]))

        
        self.image_transforms = transforms.Compose([transforms.Resize((192, 256), interpolation=transforms.InterpolationMode.NEAREST),
                   transforms.ConvertImageDtype(torch.float) 
                       ])
        

        self.mask_transforms = transforms.Compose([
            transforms.Resize((192, 256), interpolation=transforms.InterpolationMode.NEAREST)])
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        img = read_image(img_path).float() / 255.0 
        mask = read_image(mask_path, mode=ImageReadMode.GRAY)  
        
        
        img = self.image_transforms(img)
        mask = self.mask_transforms(mask).squeeze(0).long()  # Convert mask to long tensor
        
        return {"IMAGE": img, "MASK": mask}


def get_dataloader(image_dir, mask_dir, batch_size=32):
    dataset = Segmentation_Dataset(ImagesDirectory=image_dir, MasksDirectory=mask_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
