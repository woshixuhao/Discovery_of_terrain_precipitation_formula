import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import ma
import pickle
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn as nn
import torch.utils.data as data
import time
import random
import shap
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
class ANN(nn.Module):
    '''
    Construct artificial neural network
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)
        self.activation=nn.LeakyReLU()


    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

def split_dataset(X,Y,Others):
    print(X.shape,Y.shape,Others.shape)
    for name in range(5):
        if name==0:
            vmin=-1
            vmax=1
            outlier =np.where((Y[:,name]<vmin) | (Y[:,name]>vmax))[0]
            X=np.delete(X,outlier,axis=0)
            Y=np.delete(Y,outlier,axis=0)
            Others = np.delete(Others, outlier, axis=0)
        if name==1:
            vmin = -10
            vmax = 10
            outlier = np.where((Y[:, name] < vmin) | (Y[:, name] > vmax))[0]
            X = np.delete(X, outlier, axis=0)
            Y = np.delete(Y, outlier, axis=0)
            Others = np.delete(Others, outlier, axis=0)
            Y[:,name]=Y[:,name]/10
        if name == 2:
            vmin = -10
            vmax = 10
            outlier = np.where((Y[:, name] < vmin) | (Y[:, name] > vmax))[0]
            X = np.delete(X, outlier, axis=0)
            Y = np.delete(Y, outlier, axis=0)
            Others = np.delete(Others, outlier, axis=0)
            Y[:,name] = Y[:,name] / 10
        if name == 3:
            vmin = -50
            vmax = 50
            outlier = np.where((Y[:, name] < vmin) | (Y[:, name] > vmax))[0]
            X = np.delete(X, outlier, axis=0)
            Y = np.delete(Y, outlier, axis=0)
            Others = np.delete(Others, outlier, axis=0)
            Y[:,name] = Y[:,name] /10
        if name == 4:
            vmin = -10
            vmax = 10
            outlier = np.where((Y[:, name] < vmin) | (Y[:, name] > vmax))[0]
            X = np.delete(X, outlier, axis=0)
            Y = np.delete(Y, outlier, axis=0)
            Others = np.delete(Others, outlier, axis=0)
            Y[:,name] = Y[:,name] / 10
    print(X.shape,Y.shape,Others.shape)
    train_num = int(0.8 * X.shape[0])
    val_num = int(0.1 * X.shape[0])
    test_num = int(0.1 * X.shape[0])
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    np.random.set_state(state)
    np.random.shuffle(Others)
    X_train=X[0:train_num,:]
    Y_train = Y[0:train_num, :]
    Others_train = Others[0:train_num, :]
    X_validate =X[train_num:train_num + val_num, :]
    Y_validate = Y[train_num:train_num + val_num, :]
    Others_validate = Others[train_num:train_num + val_num, :]
    X_test = X[train_num + val_num:train_num + val_num + test_num, :]
    Y_test = Y[train_num + val_num:train_num + val_num + test_num, :]
    Others_test = Others[train_num + val_num:train_num + val_num + test_num, :]
    return X_train,Y_train,Others_train,X_validate,Y_validate, Others_validate,X_test,Y_test,Others_test
def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)



set_seed(1101)
device= 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(1)
file_names=os.listdir('era5_processed_data')
grid_all={}
row_all={}
variable_all=[]
variable_list = ['cape', 'cbh', 'cvh', 'cvl', 'e', 'hcc', 'i10fg', 'lai', 'lai_lv', 'lcc', 'mcc', 'msl', 'mwd',
                 'mwp', 'pev', 'si10', 'sp', 'sro', 't2m', 'tciw', 'tclw', 'u100', 'v100']
for variable_name in tqdm(variable_list):
    grid_all[variable_name]=pickle.load(open(f'era5_processed_data/grid_save_{variable_name}.pkl', 'rb'))
    row_all[variable_name] = pickle.load(open(f'era5_processed_data/variable_save_{variable_name}.pkl', 'rb'))
    variable_all.append(variable_name)
coef= pickle.load(open(f'result_save/all_coef_prediction.pkl', 'rb'))
coef_grid= pickle.load(open(f'result_save/grid_coef_prediction.pkl', 'rb'))
precip_true= pickle.load(open(f'result_save/precip_true.pkl', 'rb'))
landscape=pickle.load(open(f'result_save/each_landscape_value.pkl', 'rb'))
H=np.vstack([landscape['H'].reshape(-1,1)]*60)
R=np.vstack([landscape['R'].reshape(-1,1)]*60)
Hx=np.vstack([landscape['Hx'].reshape(-1,1)]*60)
Hy=np.vstack([landscape['Hy'].reshape(-1,1)]*60)


coef_grid = ma.masked_less(coef_grid, -1e7)

v_row_all=[]
for variable_name in tqdm(variable_all):
    v_grid=grid_all[variable_name]
    v_row = row_all[variable_name]
    v_row_all.append(v_row)


v_row_all=np.array(v_row_all)

v_row_copy=v_row_all
v_row_all=np.transpose(v_row_all[:,:,1:61],[0,2,1])
v_row_all=v_row_all.reshape(v_row_all.shape[0],v_row_all.shape[1]*v_row_all.shape[2]).T

v_max_min=np.zeros([23,2])
for i in range(23):
    v_max_min[i]=[np.max(v_row_all[:,i]),np.min(v_row_all[:,i])]
    v_row_all[:,i]=(v_row_all[:,i]-v_max_min[i,1])/(v_max_min[i,0]-v_max_min[i,1])
np.save('result_save/v_max_min.npy',v_max_min)


coef_0=np.transpose(coef[:,:,0:60],[1,2,0])
coef_0=coef_0.reshape(coef_0.shape[0],coef_0.shape[1]*coef_0.shape[2]).T

precip_true=precip_true[:,0:60].T.reshape(-1,1)

X=v_row_all
Y=coef_0
Others=np.hstack([precip_true,H,R,Hx,Hy])
print('Others',Others.shape)
X_train,Y_train,Others_train,X_validate,Y_validate,Others_validate,X_test,Y_test,Others_test=split_dataset(X,Y,Others)
X_train=torch.from_numpy(X_train.astype(np.float32))
Y_train=torch.from_numpy(Y_train.astype(np.float32))
Others_train=torch.from_numpy(Others_train.astype(np.float32))
X_validate=torch.from_numpy(X_validate.astype(np.float32))
Y_validate=torch.from_numpy(Y_validate.astype(np.float32))
Others_validate=torch.from_numpy(Others_validate.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
Y_test=torch.from_numpy(Y_test.astype(np.float32))
Others_test=torch.from_numpy(Others_test.astype(np.float32))
X_train = Variable((X_train))
y_train = Variable((Y_train))
Others_train = Variable((Others_train))
X_validate = Variable((X_validate).to(device))
y_validate = Variable((Y_validate).to(device))
Others_validate = Variable((Others_validate).to(device))
X_test = Variable((X_test).to(device))
y_test = Variable((Y_test).to(device))
Others_test = Variable((Others_test).to(device))
Net=ANN(X.shape[1],256,5).to(device)
batch_size=500000
torch_dataset = data.TensorDataset(X_train, y_train,Others_train)
loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,  # 每批提取的数量
    shuffle=True,  # 要不要打乱数据（打乱比较好）
    num_workers=0  # 多少线程来读取数据
)

optimizer = torch.optim.Adam(Net.parameters())
MODE='Train'
if MODE=='Train':
    model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    dir_name = 'model_save' + '/' + model_name
    loss_plot = []
    loss_validate_plot = []
    try:
        os.makedirs(dir_name)
    except OSError:
        pass
    with open(dir_name + '/' + 'data.txt', 'a+') as f:  # 设置文件对象
        for epoch in tqdm(range(5000)):
            for step, (batch_x, batch_y,batch_o) in enumerate(loader):
                optimizer.zero_grad()
                batch_x = Variable((batch_x).cuda(), requires_grad=True)
                batch_y=Variable((batch_y).cuda(), requires_grad=True)
                batch_o = Variable((batch_o).cuda(), requires_grad=True)
                prediction =Net(batch_x)
                precip_predict=prediction[:,0].reshape(-1,1)+\
                               prediction[:,1].reshape(-1,1)*batch_o[:,1].reshape(-1,1)*10\
                               +prediction[:,2].reshape(-1,1)*batch_o[:,2].reshape(-1,1)*10\
                               +prediction[:,3].reshape(-1,1)*batch_o[:,3].reshape(-1,1)*10\
                               +prediction[:,4].reshape(-1,1)*batch_o[:,4].reshape(-1,1)*10

                MSELoss=torch.nn.MSELoss()
                data_loss=MSELoss(batch_y, prediction)
                Precip_loss=MSELoss(batch_o[:, 0].reshape(-1,1), precip_predict)
                loss = data_loss+Precip_loss*10
                loss.backward()
                optimizer.step()

            if (epoch+1)%10==0:
                print('data loss:',data_loss.item(),'precip_loss:',Precip_loss)
                prediction_validate = Net(X_validate)
                precip_valid = prediction_validate[:, 0].reshape(-1, 1) + \
                                 prediction_validate[:, 1].reshape(-1, 1) * Others_validate[:, 1].reshape(-1, 1) * 10 \
                                 + prediction_validate[:, 2].reshape(-1, 1) *Others_validate[:, 2].reshape(-1, 1) * 10 \
                                 + prediction_validate[:, 3].reshape(-1, 1) * Others_validate[:, 3].reshape(-1, 1) * 10 \
                                 + prediction_validate[:, 4].reshape(-1, 1) * Others_validate[:, 4].reshape(-1, 1) * 10
                loss_validate = MSELoss(y_validate, prediction_validate)
                precip_loss_valid=MSELoss(Others_validate[:, 0].reshape(-1, 1), precip_valid)
                print("iter_num: %d      loss: %.8f    loss_validate: %.8f    precip_loss_validate: %.8f" % (
                    epoch + 1, loss.item(), loss_validate.item(),precip_loss_valid.item()))
                f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f  precip_loss_validate: %.8f \r\n" % (
                    epoch + 1, loss.item(), loss_validate.item(),precip_loss_valid.item()))
                torch.save(Net.state_dict(), dir_name + '/' + "net_%d_epoch.pkl" % (epoch + 1))
                torch.save(optimizer.state_dict(),
                           dir_name + '/' + "optimizer_%d_epoch.pkl" % (epoch + 1))
