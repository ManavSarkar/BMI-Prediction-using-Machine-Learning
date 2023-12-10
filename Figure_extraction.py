
# from detectron2 import detectron2
import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from Property import Body_Figure
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# from Human_Parse import HumanParser

# "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"



class Image_Processor(object):

    def __init__(self, masks_file, key_file, key_thresh=0.7):
        
        self._KeypointCfg = self.__init_key(key_file, key_thresh)
        self._KeypointsPredictor = DefaultPredictor(self._KeypointCfg)
        
        self._Contourcfg=self.__init_mask(masks_file,key_thresh)
        self._ContourPredictor = DefaultPredictor(self._Contourcfg)
        
        # self._HumanParser = HumanParser()
        
    def __init_key(self, key_file, key_thresh):
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(key_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = key_thresh  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(key_file)
        
        return cfg
    
    def __init_mask(self, mask_file, key_thresh):
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(mask_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = key_thresh  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(mask_file)
        return cfg


    def get_keyandcontour_output(self, img):
        
        Keypoints=self._Keypoints_detected(img)
        
        ContourOutput=self._Contour_detected(img)

        # """ Detect Arms Mask by Human parser """
        # Arms_mask = self._HumanParser.Arms_detect(img)
        # ContourOutput = ContourOutput ^ Arms_mask

        return Keypoints, ContourOutput

    def _Contour_detected(self,img):
    
        ContourOutput=self._ContourPredictor(img)
        sorted_idxs = np.argsort(-ContourOutput["instances"].scores.cpu().numpy())
        ContourMasks = None
        for sorted_idx in sorted_idxs:
            if ContourOutput["instances"].pred_classes[sorted_idx] == 0:
                ContourMasks = ContourOutput["instances"].pred_masks[sorted_idx].cpu().numpy()
                
        ContourOutput = ContourMasks
        return ContourOutput
        

    def _Keypoints_detected(self,img):
        
        KeypointsOutput = self._KeypointsPredictor(img)
        sorted_idxs = np.argsort(-KeypointsOutput["instances"].scores.cpu().numpy())
        Keypoints = KeypointsOutput["instances"].pred_keypoints[sorted_idxs[0]].cpu().numpy()
        
        return Keypoints

    # def Process(self, img_RGB):
    def get_figure(self, img):
        Keypoints, ContourOutput = self.get_keyandcontour_output(img)
        
        nose,left_ear,right_ear,left_shoulder,right_shoulder = Keypoints[0],Keypoints[4],Keypoints[3],Keypoints[6], Keypoints[5]
        
        left_hip, right_hip, left_knee, right_knee = Keypoints[12], Keypoints[11], Keypoints[14],Keypoints[13]
        
        y_hip = (left_hip[1] + right_hip[1]) / 2
        y_knee = (left_knee[1] + right_knee[1]) / 2

        center_shoulder = (left_shoulder + right_shoulder) / 2
        
        y_waist = y_hip * 2 / 3 + (nose[1] + center_shoulder[1]) / 6
        
        left_thigh = (left_knee + left_hip) / 2
        right_thigh = (right_knee + right_hip) / 2

        # estimate the waist width
        waist_width = self.waist_width_estimate(center_shoulder, y_waist, ContourOutput)
        
        # estimate the thigh width
        thigh_width = self.thigh_width_estimate(left_thigh, right_thigh, ContourOutput)
        
        # estimate the hip width
        hip_width = self.hip_width_estimate(center_shoulder, y_hip, ContourOutput)
        
        # estimate the head_width
        head_width = self.head_width_estimate(left_ear, right_ear)
        
        # estimate the Area
        Area = self.Area_estimate(y_waist, y_hip, waist_width, hip_width, ContourOutput)
        
        # estimate the height2waist
        height = self.Height_estimate(y_knee, nose[1])
        
        # estimate tht shoulder_width
        shoulder_width = self.shoulder_width_estimate(left_shoulder, right_shoulder)

        figure = Body_Figure(waist_width, thigh_width, hip_width, head_width, Area, height, shoulder_width)
        
#         outputs = self._KeypointsPredictor(img)
#         v = Visualizer(img[:,:,::-1], MetadataCatalog.get( self._KeypointCfg.DATASETS.TRAIN[0]), scale=1.2)
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])
#         cv2.imwrite('random.jpg', out.get_image()[:, :, ::-1])
        
#         outputs = self._ContourPredictor(img)
#         v = Visualizer(img[:,:,::-1], MetadataCatalog.get( self._Contourcfg.DATASETS.TRAIN[0]), scale=1.2)
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # cv2_imshow(out.get_image()[:, :, ::-1])
#         cv2.imwrite('random1.jpg', out.get_image()[:, :, ::-1])

        return figure

    def Height_estimate(self, y_k, y_n):
        Height = np.abs(y_n - y_k)
        return Height

    def Area_estimate(self, y_w, y_h, W_w, H_w, mask):
        # '''
        #     Area is expressed as thenumber of
        #     pixels per unit area between waist and hip
        # '''
        try:
            pixels = np.sum(mask[int(y_w):int(y_h)][:])
        except:
            pixels=100
        
        area = (y_h - y_w) * 0.5 * (W_w + H_w)
        Area = pixels / area
        return Area

    def shoulder_width_estimate(self, left_shoulder, right_shoulder):
        shoulder_width = np.sqrt((right_shoulder[0] - left_shoulder[0]) ** 2 + (right_shoulder[1] - left_shoulder[1]) ** 2)
        return shoulder_width

    def head_width_estimate(self, left_ear, right_eat):
        head_width = np.sqrt((right_eat[0] - left_ear[0]) ** 2 + (right_eat[1] - left_ear[1]) ** 2)
        return head_width

    def hip_width_estimate(self, center_shoulder, y_hip, ContourOutput):
        x_hip_center = int(center_shoulder[0])
        try:
            x_lhb = np.where(ContourOutput[int(y_hip)][:x_hip_center] == 0)[0]
            x_lhb = x_lhb[-1] if len(x_lhb) else 0
        except:
            x_lhb = 10
        try:
            x_rhb = np.where(ContourOutput[int(y_hip)][x_hip_center:] == 0)[0]
            x_rhb = x_rhb[0] + x_hip_center if len(x_rhb) else len(ContourOutput[0])
        except:
            x_rhb = 5
        hip_width = x_rhb - x_lhb
        return hip_width

    def thigh_width_estimate(self, left_thigh, right_thigh, mask):
        lx, ly = int(left_thigh[0]), int(left_thigh[1])
        rx, ry = int(right_thigh[0]), int(right_thigh[1])
        try:
            x_ltb = np.where(mask[ly][:lx] == 0)[0]
            x_ltb = x_ltb[-1] if len(x_ltb) else 0
        except:
            x_ltb = 10
        try:
            
            x_rtb = np.where(mask[ry][rx:] == 0)[0]
            x_rtb = x_rtb[0] + rx if len(x_rtb) else len(mask[0])
        except:
            x_rtb = 0
        l_width = (lx - x_ltb) * 2
        r_width = (x_rtb - rx) * 2

        thigh_width = (l_width + r_width) / 2
        return thigh_width

    def waist_width_estimate(self, center_shoulder, y_waist, ContourOutput):
        x_waist_center = int(center_shoulder[0])
        # plt.imshow(ContourOutput)
        # plt.show()
        try:
            x_lwb = np.where(ContourOutput[int(y_waist)][:x_waist_center] == 0)[0]
            x_lwb = x_lwb[-1] if len(x_lwb) else 0
        except:
            x_lwb = 10
            print("err waist width")
        try:
            x_rwb = np.where(ContourOutput[int(y_waist)][x_waist_center:] == 0)[0]
            x_rwb = x_rwb[0] + x_waist_center if len(x_rwb) else len(ContourOutput[0])
        except:
            x_rwb=0
            print("err waist width")
        # print(x_rwb)
        waist_width = x_rwb - x_lwb
        return waist_width

