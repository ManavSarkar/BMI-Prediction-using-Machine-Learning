import numpy as np
import pandas
import cv2
from PIL import Image
from Figure_extraction import Image_Processor
import torchvision.models.detection
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights



class Data_Processor(object):
    
    def __init__(self,mask_model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        keypoints_model = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"):
        
        
        self._img_pro = Image_Processor(mask_model,keypoints_model)
        

    def get_image_info(self,df):
        return df
    
    def test(self,img_path):
        img = cv2.imread(img_path)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # print(img)
        # img_path = '/kaggle/input/weightpred/datasets/Images/002274_M_35_175260_12337713.jpg'
        # img = Image.open(img_path)
        figure = self._img_pro.get_figure(img)
        return figure
        
# print(Image.open("Images/0_F_18_162560_6531731.jpg"))

# obj.test("project_images/0_F_18_162560_6531731.jpg")
# obj=Data_Processor()

# obj.test("../targetdir/project_images/0_F_15_162560_6123497.jpg")




