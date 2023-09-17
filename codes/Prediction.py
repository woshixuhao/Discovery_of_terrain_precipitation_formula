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
from plot_geo import *
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
v_row_copy=v_row_all.copy()
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
batch_size=200000
torch_dataset = data.TensorDataset(X_train, y_train,Others_train)
loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,  # 每批提取的数量
    shuffle=True,  # 要不要打乱数据（打乱比较好）
    num_workers=0  # 多少线程来读取数据
)

optimizer = torch.optim.Adam(Net.parameters())
MODE='shap'

if MODE=='Prediction':
    # print(variable_all)
    # print(variable_all[23])
    # v_row_mean=np.mean(v_row_copy,axis=2).T
    # print(v_row_mean[23,0:50])
    # print(v_row_mean.shape)
    ssp_num=585
    influence_factor='lai'
    landscape = pickle.load(open(f'result_save/each_landscape_value.pkl', 'rb'))
    H = landscape['H'].reshape(-1, )
    R = landscape['R'].reshape(-1, )
    Hx = landscape['Hx'].reshape(-1, )
    Hy = landscape['Hy'].reshape(-1, )
    t2m = pickle.load(open(f'CMIP6_processed/variable_save_{influence_factor}_ssp{ssp_num}.pkl', 'rb'))
    t2m_grid = pickle.load(open(f'CMIP6_processed/grid_save_{influence_factor}_ssp{ssp_num}.pkl', 'rb'))
    t2m_history = pickle.load(open(f'CMIP6_processed/variable_save_{influence_factor}_historical.pkl', 'rb'))
    t2m_grid_history = pickle.load(open(f'CMIP6_processed/grid_save_{influence_factor}_historical.pkl', 'rb'))
    lai_ratio_row= pickle.load(open(f'result_save/lai_ratio_row.pkl', 'rb'))
    with open('result_save/coord_save.pkl', 'rb') as f:
        coord_save = pickle.load(f)
    print(t2m.shape)

    best_epoch = 4000
    Net_0 = ANN(X.shape[1], 256, 5).to(device)
    Net_0.load_state_dict(torch.load(f'model_save/09-08-21-30/Net_{best_epoch}_epoch.pkl'))
    v_row_copy = np.transpose(v_row_copy, [0, 2, 1])

    mean_precip_2100=ma.ones([184,288,30])*(-1e8)
    mean_precip_2015=ma.ones([184,288,30])*(-1e8)
    index_m=0
    for m in range(135, 165):
        X = v_row_copy[:, 66, :].T
        if influence_factor=='ts':
            X[:,18] = t2m_history[:, m]
        if influence_factor == 'lai':
            X[:, 7] = t2m_history[:, m]
        X_0 = X.copy()

        v_max_min = np.load('result_save/v_max_min.npy')
        for i in range(23):
            X_0[:, i] = (X_0[:, i] - v_max_min[i, 1]) / (v_max_min[i, 0] - v_max_min[i, 1])

        print(X_0[0])
        X_0 = Variable(torch.from_numpy(X_0.astype(np.float32)).to(device))

        y_0 = Net_0(X_0)[:, 0].cpu().data.numpy()
        y_1 = Net_0(X_0)[:, 1].cpu().data.numpy() * 10
        y_2 = Net_0(X_0)[:, 2].cpu().data.numpy() * 10
        y_3 = Net_0(X_0)[:, 3].cpu().data.numpy() * 10
        y_4 = Net_0(X_0)[:, 4].cpu().data.numpy() * 10

        predict_precio = y_0 + y_1 * H + y_2 * R + y_3 * Hx + y_4 * Hy

        print(predict_precio.shape)
        all_precip = ma.ones([184, 288]) * (-1e8)
        index_num = 0
        for coord in coord_save:
            all_precip[coord[0], coord[1]] = predict_precio[index_num]
            index_num += 1
        all_precip = ma.masked_less(all_precip, -1e7)
        all_precip = ma.masked_greater(all_precip, 1)
        all_precip = all_precip * (10850 - 1.7107198) + 1.7107198
        all_precip[all_precip < 1e-8] = 1e-8
        all_precip_2 = all_precip
        mean_precip_2015[:, :, index_m] = all_precip_2
        index_m += 1
    index_m = 0
    for m in range(66,86):
        X = v_row_copy[:, 66, :].T
        if influence_factor == 'ts':
            X[:, 18] = t2m[:, m]
        if influence_factor == 'lai':
            X[:, 7] = t2m[:, m]
        X_0 = X.copy()

        v_max_min = np.load('result_save/v_max_min.npy')
        for i in range(23):
            X_0[:, i] = (X_0[:, i] - v_max_min[i, 1]) / (v_max_min[i, 0] - v_max_min[i, 1])

        print(X_0[0])
        X_0 = Variable(torch.from_numpy(X_0.astype(np.float32)).to(device))

        y_0 = Net_0(X_0)[:, 0].cpu().data.numpy()
        y_1 = Net_0(X_0)[:, 1].cpu().data.numpy() * 10
        y_2 = Net_0(X_0)[:, 2].cpu().data.numpy() * 10
        y_3 = Net_0(X_0)[:, 3].cpu().data.numpy() * 10
        y_4 = Net_0(X_0)[:, 4].cpu().data.numpy() * 10

        predict_precio = y_0 + y_1 * H + y_2 * R + y_3 * Hx + y_4 * Hy

        print(predict_precio.shape)
        all_precip = ma.ones([184, 288]) * (-1e8)
        index_num = 0
        for coord in coord_save:
            all_precip[coord[0], coord[1]] = predict_precio[index_num]
            index_num += 1
        all_precip = ma.masked_less(all_precip, -1e7)
        all_precip = ma.masked_greater(all_precip, 1)
        all_precip = all_precip * (10850 - 1.7107198) + 1.7107198
        all_precip[all_precip < 1e-8] = 1e-8
        all_precip_1 = all_precip
        mean_precip_2100[:,:,index_m]=all_precip_1
        index_m+=1
    pickle.dump(mean_precip_2015, open(fr'result_save/{influence_factor}_mean_precip_2015_history.pkl', 'wb'), protocol=4)
    pickle.dump(mean_precip_2100, open(fr'result_save/{influence_factor}_mean_precip_2100_{ssp_num}.pkl', 'wb'), protocol=4)

if MODE=='valid_predict':
    v_row_copy = np.transpose(v_row_copy [:, :, 61:], [0, 2, 1])
    coef_0 = np.transpose(coef[:, :, 60:], [1, 2, 0])
    print(v_row_copy.shape)
    for i_year in range(4):
        v_row_valid =v_row_copy[:, i_year,:].T
        coef=coef_0[:, i_year,:].T

        X_validate=v_row_valid
        Y_validate=coef
        print(X_validate.shape, Y_validate.shape)


        landscape = pickle.load(open(f'result_save/each_landscape_value.pkl', 'rb'))
        H =landscape['H'].reshape(-1, )
        R = landscape['R'].reshape(-1, )
        Hx = landscape['Hx'].reshape(-1,)
        Hy = landscape['Hy'].reshape(-1,)
        with open('result_save/coord_save.pkl', 'rb') as f:
            coord_save = pickle.load(f)

            best_epoch = 4000
            Net_0 = ANN(X.shape[1], 256, 5).to(device)
            Net_0.load_state_dict(torch.load(f'model_save/Net_{best_epoch}_epoch.pkl'))


        X = X_validate
        X_0 = X.copy()
        v_max_min=np.load('result_save/v_max_min.npy')
        for i in range(23):
            X_0[:, i] = (X_0[:, i] - v_max_min[i, 1]) / (v_max_min[i, 0] - v_max_min[i, 1])

        print(X_0[0])
        X_0 = Variable(torch.from_numpy(X_0.astype(np.float32)).to(device))

        y_0 = Net_0(X_0)[:,0].cpu().data.numpy()
        y_1 = Net_0(X_0)[:,1].cpu().data.numpy()*10
        y_2 = Net_0(X_0)[:,2].cpu().data.numpy()*10
        y_3 = Net_0(X_0)[:,3].cpu().data.numpy()*10
        y_4 = Net_0(X_0)[:,4].cpu().data.numpy()*10

        # for h in range(5):
        #     plt.scatter(Net_0(X_0)[:,h].cpu().data.numpy(),Net_0(X_0)[:,h].cpu().data.numpy())
        #     plt.show()

        predict_precio = y_0 + y_1 * H + y_2 * R + y_3 * Hx + y_4 * Hy

        print(predict_precio.shape)
        all_precip = ma.ones([184, 288]) * (-1e8)
        index_num = 0
        for coord in coord_save:
            all_precip[coord[0], coord[1]] = predict_precio[index_num]
            index_num += 1
        all_precip = ma.masked_less(all_precip, -1e7)
        all_precip = ma.masked_greater(all_precip, 1)
        all_precip = all_precip * (10850 - 1.7107198) + 1.7107198
        all_precip[all_precip<1e-8]=1e-8
        #plot_geo(all_precip,f'new_prediction_precip_{i_year + 2011}')
        pickle.dump(all_precip, open(fr'result_save/all_precip_{i_year + 2011}.pkl', 'wb'), protocol=4)

if MODE=='shap':
    Net_0 = ANN(X.shape[1], 256, 5).to(device)
    best_epoch = 4000
    Net_0.load_state_dict(torch.load(f'model_save/09-08-21-30/Net_{best_epoch}_epoch.pkl'))
    Net.eval()

    example_num=1000
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = 'Arial'
    explain_data = X_test[np.random.choice(X_test.shape[0], example_num, replace=False)][0:example_num]
    explainer = shap.DeepExplainer(Net, X_train[np.random.choice(X_train.shape[0], example_num, replace=False)][
                                        0:example_num].to(device))
    shap_values = explainer.shap_values(explain_data)
    for coef_index in range(5):
        shap.summary_plot(shap_values[coef_index], explain_data.cpu().data.numpy(),feature_names=variable_all,
                          plot_type="bar",color='#808080',max_display=6, show = False)
        shap.summary_plot(shap_values[coef_index], explain_data.cpu().data.numpy(), feature_names=variable_all, max_display=6,alpha=0.3, show=False)
        plt.savefig(f'plot_save/shap/summary_plot_{coef_index}.pdf')
        plt.savefig(f'plot_save/shap/summary_plot_{coef_index}.jpg')
        plt.show()







