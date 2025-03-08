from PIL import Image, ImageOps
from torchvision import transforms

class ScaleAndPadTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def transform(self, img):
        width, height = img.size
        if width > height:
            scale = self.target_size / width
            new_height = int(height * scale)
            img = img.resize((self.target_size, new_height))
            padding = (self.target_size - new_height) // 2
            img = ImageOps.expand(img, (0, padding, 0, self.target_size - new_height - padding))
        else:
            scale = self.target_size / height
            new_width = int(width * scale)
            img = img.resize((new_width, self.target_size))
            padding = (self.target_size - new_width) // 2
            img = ImageOps.expand(img, (padding, 0, self.target_size - new_width - padding, 0))

        
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.CenterCrop(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])
        img = transform(img)

        return img

