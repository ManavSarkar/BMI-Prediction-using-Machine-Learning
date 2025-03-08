from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
from scale_transform import ScaleAndPadTransform
class CustomDataset(Dataset):
    def __init__(self, dataset,img_folder, transform=None):
        self.data = dataset
        self.transform = transform
        self.img_folder = img_folder
        self.transform = ScaleAndPadTransform(224).transform

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] 
        img_path = self.img_folder + img_name  # adjust the path to your actual image directory
        image = Image.open(img_path)
        
        ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
        BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
        
        if self.transform:
            image = self.transform(image)
        
        return (image,img_name), BMI


def get_dataloader(df,img_folder= "datasets/Images/", batch_size=64, shuffle=True, num_workers=4):
    train_data = df[df['split'] == 'train']
    test_data = df[df['split'] == 'test']
    val_data = df[df['split'] == 'validation']

    train_dataset = CustomDataset(train_data, img_folder= img_folder)
    test_dataset = CustomDataset(test_data,img_folder= img_folder)
    val_dataset = CustomDataset(val_data,img_folder= img_folder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader, val_loader