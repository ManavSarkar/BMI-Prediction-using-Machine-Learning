from sklearn.metrics import mean_absolute_error
import re
import csv
from custom_resnet import custom_resnet
from dataset_loader import get_dataloader
import pandas as pd
import torch
from Data_extraction import Data_Processor
df = pd.read_csv('datasets/updated_data.csv')
resnet_model = custom_resnet()
obj=Data_Processor()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def fix_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("fc1.") and not key.startswith("fc1.0"):
            new_key = key.replace("fc1.", "fc1.0.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Load the state dictionary with modified keys
state_dict = torch.load('model_epoch_50.ckpt', map_location=device)['state_dict']
fixed_state_dict = fix_state_dict_keys(state_dict)


# resnet_model = DFNet(df=15, bf=0)
resnet_model.load_state_dict(state_dict)
resnet_model = resnet_model.to(device)
resnet_model.eval()


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()
        
        
loaders = get_dataloader(df=df, batch_size=1)
files = ['train', 'test', 'validation']


for loader, file in zip(loaders, files):
    cnt = 0
    if(file=='train'):
        continue
    with open(
            'Image_{}.csv'.format(file),
            'a+', newline='') as fp:
        writer = csv.writer(fp)
        pred = []
        targ = []
        for (data, img_name), target in loader:
            cnt += 1
            print(cnt)
            values = []
            data, target = data.to(device), target.to(device)
#             print(img_name[0])
            ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name[0])
            
            values.append(img_name[0])
            values.append(target.cpu().numpy()[0])
            
            sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
            values.append(sex)
            body_feature = obj.test("datasets/Images/"+img_name[0])



            values.append(body_feature.WSR)
            values.append(body_feature.WTR)
            values.append(body_feature.WHpR)
            values.append(body_feature.WHdR)
            values.append(body_feature.HpHdR)
            values.append(body_feature.Area)
            values.append(body_feature.H2W)

            conv_out = LayerActivations(resnet_model.fc1, None)
            out = resnet_model(data)
            pred.append(out.item())
            targ.append(target.item())
            conv_out.remove()
            xs = torch.squeeze(conv_out.features.cpu().detach()).numpy()

            for x in xs:
                values.append(float(x))
            
            age = int(ret.group(2))
            height = int(ret.group(3)) / 100000
            weight = int(ret.group(4)) / 100000
            
            values.append(age)
            values.append(height)
            values.append(weight)

            writer.writerow(values)
        MAE = mean_absolute_error(targ, pred)
        print(file, ' ', MAE)