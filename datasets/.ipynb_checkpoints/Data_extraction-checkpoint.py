import numpy as np
import pandas
import cv2
from PIL import Image
from Figure_extraction import Image_Processor
import torchvision.models.detection
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torch

class Data_Processor(object):
    
    def __init__(self):
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mask_model = maskrcnn_resnet50_fpn(weights=weights, progress=False).to(device)
        mask_model = mask_model.eval()
        key_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        # call the eval() method to prepare the model for inference mode.
        key_model.eval()
        
        self._img_pro = Image_Processor("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml","COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        

    def get_image_info(self,df):
        # all_images=df['image']
        # # columns=['WTR','WHpR','WHdR','HpHdR','Area','H2W','WSR','sex','age','height','weight','BMI']
        # for img_path in all_images:
        #         img = cv2.imread(img_path)
        #         figure = self._img_pro.get_figure(img)
        #         print(figure)
        
       

           
        return df
    
    def test(self,img_path):
        # img = cv2.imread(img_path)
        # img_path = '/kaggle/input/weightpred/datasets/Images/002274_M_35_175260_12337713.jpg'
        img = cv2.imread(img_path)
        figure = self._img_pro.Process(img)
        print(figure.WSR())

print(Image.open("Images/0_F_18_162560_6531731.jpg"))
obj=Data_Processor()

obj.test("Images/0_F_18_162560_6531731.jpg")




