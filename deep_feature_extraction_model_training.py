from custom_resnet import custom_resnet, custom_resnet_optimizer
from dataset_loader import get_dataloader
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import mean_absolute_error
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

resnet_model = custom_resnet()
optimizer = custom_resnet_optimizer(resnet_model)


df = pd.read_csv('datasets/updated_data.csv')

train_loader, test_loader, val_loader = get_dataloader( df=df)


resnet_model = resnet_model.to(device)
criterion = nn.MSELoss()


def train(model, device, train_loader, epoch):
    model.train()
    running_loss = 0.0
    for idx, ((x,n),y) in enumerate(train_loader,0):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
#         print(y_pred.shape)
        y = torch.unsqueeze(y,1)
        loss = criterion(y_pred.double(), y.double())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print('loss: ', loss.item())
    print('Train Epoch:{}\t RealLoss:{:.6f}'.format(epoch, running_loss / len(train_loader)))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def test(model, device, test_loader):
    model.eval()
    pred = []
    targ = []
    with torch.no_grad():
        for i, ((x, n), y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            # optimizer.zero_grad()
            y_pred = model(x)
            # print(y_pred.shape)
            pred.append(y_pred.item())
            targ.append(y.item())
            y = torch.unsqueeze(y, 1)
    MAE = mean_absolute_error(targ, pred)
    MAPE = mean_absolute_percentage_error(targ, pred)
    print('\nTest MAE:{}\t Test MAPE:{} '.format(MAE, MAPE))
    return MAE, MAPE

for epoch in range(100):
    print('*' * 50)
    train(resnet_model, device, train_loader, epoch)
    val_MAE, val_MAPE = test(resnet_model, device, val_loader)
    if val_MAE < MIN_MAE:
        MIN_MAE = val_MAE
        torch.save(resnet_model.state_dict(), 'MIN_RESNET101_BMI_Cache_test.pkl')
        END_EPOCH = epoch


Net = resnet_model
Net.load_state_dict(torch.load('MIN_RESNET101_BMI_Cache_test.pkl'))
Net = Net.to(device)
print('=' * 50)
Net.eval()
test(Net, device, test_loader)

print('END_EPOCH:', END_EPOCH)
print('=' * 50)