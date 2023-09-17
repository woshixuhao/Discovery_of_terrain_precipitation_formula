import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import ma
import pickle
import os
import math
'''
cltc:CMIP6.CMIP.NCAR.CESM2.historical.r7i1p1f1.AERmon.cltc.gn

'''

def read_nc_aphro(nc_obj,print_info=False):
    '''
    2008年之前的数据是netCDF可以读取的，之后的由read_h5py_2008读取
    APHRO Includes variable: 'time', 'longitude', 'latitude', 'precip', 'rstn'
    each is 365*184*288
    longitude:经度
    latitude:纬度
    precip：降水
    rstn:射电太阳望远镜网络
    '''
    if print_info==True:
        #查看nc文件有些啥东东
        print(nc_obj)
        print('---------------------------------------')

        #查看nc文件中的变量
        print(nc_obj.variables.keys())
        for i in nc_obj.variables.keys():
            print(i)
        print('---------------------------------------')

    data={}
    variable_list=[]
    for i in nc_obj.variables.keys():
        variable_list.append(i)
        data[i]=nc_obj.variables[i][:]

    return variable_list,data

def save_dataset(variable_name):
    filePath = fr'CMIP6_origin_data/' + variable_name + '/'
    file_names = os.listdir(filePath)
    try:
        os.makedirs(fr'CMIP6_data/' + variable_name + '/')
    except OSError:
        pass
    for file_name in file_names:
        full_file_name = filePath + file_name
        nc_obj = netCDF4.Dataset(full_file_name)
        variable_list,data = read_nc_aphro(nc_obj, print_info=False)
        pickle.dump(data, open(fr'CMIP6_data/' + variable_name + '/'+file_name[0:-4]+'.pkl', 'wb'), protocol=4)

def load_dataset(file_path):
    filePath = fr'CMIP6_data/' + variable_name + '/'
    file_names = os.listdir(filePath)
    index=0
    for file in file_names:
        if index==0:
            data=pickle.load(open(filePath+file, 'rb'))
        else:
            new_data=pickle.load(open(filePath+file, 'rb'))
            for key in data.keys():
                if key==variable_name:
                    data[key]=ma.concatenate((data[key],new_data[key]),axis=0)
        index+=1

    return data

def get_year_data(data,variable_name):
    # data = load_dataset(variable_name)
    lat_loc=np.where((data['lat'] <= 54.875) & (data['lat'] >= 9.125))[0]
    lon_loc=np.where((data['lon'] <= 140.875) & (data['lon'] > 69.125))[0]

    # if variable_name=='ts':
    #     #1850~2014
    data_month=data[variable_name][:]
    year_num=int(data_month.shape[0]/12)
    for i in range(year_num):
        loc=ma.mean(data_month[12*i:12*(i+1)],axis=0).reshape(1,data_month.shape[1],data_month.shape[2])
        if i==0:
            data_year=loc
        else:
            data_year=ma.concatenate((data_year,loc),axis=0)

    data[variable_name]=data_year[:,np.min(lat_loc):(np.max(lat_loc)+1),np.min(lon_loc):(np.max(lon_loc)+1)]
    data['lon']=data['lon'][np.min(lon_loc):(np.max(lon_loc)+1)]
    data['lat'] = data['lat'][np.min(lat_loc):(np.max(lat_loc) + 1)]
    print(data['lon'].shape, data['lat'].shape)
    return data

def find_nearest(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return idx

def match_coord(data,variable_name,mode):
    with open('result_save/coord_save.pkl', 'rb') as f:
        coord_save = pickle.load(f)
    Dataset_1951_2015 = pickle.load(open(f'Dataset/Dataset_1951.pkl', 'rb'))
    lat = Dataset_1951_2015['latitude']
    lon = Dataset_1951_2015['longitude']
    variable_save=[]
    grid_save = ma.ones([data[variable_name].shape[0], lat.shape[0], lon.shape[0]]) * (-1e8)
    for coord in tqdm(coord_save):
        lat_idx=find_nearest(data['lat'],lat[coord[0]])
        lon_idx=find_nearest(data['lon'],lon[coord[1]])
        variable_save.append(data[variable_name][:,lat_idx,lon_idx])
        grid_save[:,coord[0],coord[1]]=data[variable_name][:,lat_idx,lon_idx]
    grid_save = ma.masked_less(grid_save, -1e7)
    variable_save=np.vstack(variable_save)
    pickle.dump(variable_save, open(fr'CMIP6_processed/variable_save_{variable_name}_{mode}.pkl', 'wb'), protocol=4)
    pickle.dump(grid_save, open(fr'CMIP6_processed/grid_save_{variable_name}_{mode}.pkl', 'wb'), protocol=4)
    return variable_save,grid_save
def standardlization(x):
    return (x - ma.min(x)) / (ma.max(x) - ma.min(x))
def standardlization_aspect(aspect):
    for i in range(aspect.shape[0]):
        for j in range(aspect.shape[1]):
            if aspect[i,j]<=180:
                aspect[i,j]=aspect[i,j] / 180 * math.pi
            if aspect[i,j]>180:
                aspect[i,j]=(360-aspect[i,j])/ 180 * math.pi
            return aspect
def FiniteDiff_x(u, dx, d):
    # 用二阶微分计算d阶微分，不过在三阶以上准确性会比较低
    # u是需要被微分的数据
    # dx是网格的空间大小
    nt, nx = u.shape
    ux = np.zeros([nt, nx])

    if d == 1:
        ux[:, 1:nx - 1] = (u[:, 2:nx] - u[:, 0:nx - 2]) / (2 * dx)
        ux[:, 0] = (-3.0 / 2 * u[:, 0] + 2 * u[:, 1] - u[:, 2] / 2) / dx
        ux[:, nx - 1] = (2.0 / 2 * u[:, nx - 1] - 2 * u[:, nx - 2] + u[:, nx - 3] / 2) / dx
        return ux

    if d == 2:
        ux[:, 1:nx - 1] = (u[:, 2:nx] - 2 * u[:, 1:nx - 1] + u[:, 0:nx - 2]) / dx ** 2
        ux[:, 0] = (2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]) / dx ** 2
        ux[:, nx - 1] = (2 * u[:, nx - 1] - 5 * u[:, nx - 2] + 4 * u[:, nx - 3] - u[:, nx - 4]) / dx ** 2
        return ux

    if d == 3:
        ux[:, 2:nx - 2] = (u[:, 4:nx] / 2 - u[:, 3:nx - 1] + u[:, 1:nx - 3] - u[:, 0:nx - 4] / 2) / dx ** 3
        ux[:, 0] = (-2.5 * u[:, 0] + 9 * u[:, 1] - 12 * u[:, 2] + 7 * u[:, 3] - 1.5 * u[:, 4]) / dx ** 3
        ux[:, 1] = (-2.5 * u[:, 1] + 9 * u[:, 2] - 12 * u[:, 3] + 7 * u[:, 4] - 1.5 * u[:, 5]) / dx ** 3
        ux[:, nx - 1] = (2.5 * u[:, nx - 1] - 9 * u[:, nx - 2] + 12 * u[:, nx - 3] - 7 * u[:, nx - 4] + 1.5 * u[:,
                                                                                                              nx - 5]) / dx ** 3
        ux[:, nx - 2] = (2.5 * u[:, nx - 2] - 9 * u[:, nx - 3] + 12 * u[:, nx - 4] - 7 * u[:, nx - 5] + 1.5 * u[:,
                                                                                                              nx - 6]) / dx ** 3
        return ux

    if d > 3:
        return FiniteDiff_x(FiniteDiff_x(u, dx, 3), dx, d - 3)
def FiniteDiff_y(un, dx, d):
    # u=[nx,nt]
    # 用二阶微分计算d阶微分，不过在三阶以上准确性会比较低
    # u是需要被微分的数据
    # dx是网格的空间大小
    u = un
    nx, nt = u.shape
    ux = np.zeros([nx, nt])

    if d == 1:
        ux[:, 1:nt - 1] = (u[:, 2:nt] - u[:, 0:nt - 2]) / (2 * dx)
        ux[:, 0] = (-3.0 / 2 * u[:, 0] + 2 * u[:, 1] - u[:, 2] / 2) / dx
        ux[:, nt - 1] = (2.0 / 2 * u[:, nt - 1] - 2 * u[:, nt - 2] + u[:, nt - 3] / 2) / dx
        return ux

    if d == 2:
        ux[:, 1:nt - 1] = (u[:, 2:nt] - 2 * u[:, 1:nt - 1] + u[:, 0:nt - 2]) / dx ** 2
        ux[:, 0] = (2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]) / dx ** 2
        ux[:, nt - 1] = (2 * u[:, nt - 1] - 5 * u[:, nt - 2] + 4 * u[:, nt - 3] - u[:, nt - 4]) / dx ** 2
        return ux

    if d == 3:
        ux[:, 2:nt - 2] = (u[:, 4:nt] / 2 - u[:, 3:nt - 1] + u[:, 1:nt - 3] - u[:, 0:nt - 4] / 2) / dx ** 3
        ux[:, 0] = (-2.5 * u[:, 0] + 9 * u[:, 1] - 12 * u[:, 2] + 7 * u[:, 3] - 1.5 * u[:, 4]) / dx ** 3
        ux[:, 1] = (-2.5 * u[:, 1] + 9 * u[:, 2] - 12 * u[:, 3] + 7 * u[:, 4] - 1.5 * u[:, 5]) / dx ** 3
        ux[:, nt - 1] = (2.5 * u[:, nt - 1] - 9 * u[:, nt - 2] + 12 * u[:, nt - 3] - 7 * u[:, nt - 4] + 1.5 * u[:,
                                                                                                              nt - 5]) / dx ** 3
        ux[:, nt - 2] = (2.5 * u[:, nt - 2] - 9 * u[:, nt - 3] + 12 * u[:, nt - 4] - 7 * u[:, nt - 5] + 1.5 * u[:,
                                                                                                              nt - 6]) / dx ** 3
        return ux

    if d > 3:
        return FiniteDiff_y(FiniteDiff_y(u, dx, 3), dx, d - 3)
def get_landsacpe_variable():
    with open('result_save/coord_save.pkl', 'rb') as f:
        coord_save = pickle.load(f)
    Dataset_1951_2015 = pickle.load(open(f'Dataset/Dataset_1951.pkl', 'rb'))
    lat = Dataset_1951_2015['latitude']
    lon = Dataset_1951_2015['longitude']
    HT = Dataset_1951_2015['HT'][0]  # shape(184,288)
    aspect = Dataset_1951_2015['spect']
    relief = Dataset_1951_2015['relief']
    slope = Dataset_1951_2015['slope']
    lat = Dataset_1951_2015['latitude']
    lon = Dataset_1951_2015['longitude']

    wl = 3  # window length
    large_HT = ma.max(HT)
    small_HT = ma.min(HT)
    large_relief = ma.max(relief)
    small_relief = ma.min(relief)
    # ========Standardlization==================
    HT = standardlization(HT)
    aspect = standardlization_aspect(aspect)
    relief = standardlization(relief)
    slope = slope / 180 * math.pi
    dHTdx = FiniteDiff_x(HT, 1, d=1)
    dHTdy = FiniteDiff_y(HT.T, 1, d=1).T
    all_HT=[]
    all_aspect=[]
    all_relief=[]
    all_slope=[]
    all_dHx=[]
    all_dHy=[]
    for coord in tqdm(coord_save):
        all_HT.append(HT[coord[0],coord[1]])
        all_aspect.append(aspect[coord[0], coord[1]])
        all_relief.append(relief[coord[0], coord[1]])
        all_slope.append(slope[coord[0], coord[1]])
        all_dHx.append(dHTdx[coord[0], coord[1]])
        all_dHy.append(dHTdy[coord[0], coord[1]])
    all_HT=np.array(all_HT).reshape(-1,1)
    all_aspect = np.array(all_aspect).reshape(-1, 1)
    all_relief = np.array(all_relief).reshape(-1, 1)
    all_slope = np.array(all_slope).reshape(-1, 1)
    all_dHx = np.array(all_dHx).reshape(-1, 1)
    all_dHy = np.array(all_dHy).reshape(-1, 1)

    data={'HT':all_HT,'aspect':all_aspect,'relief':all_relief,'slope':all_slope,'dHx':all_dHx,'dHy':all_dHy}
    pickle.dump(data, open(fr'result_save/landscape.pkl', 'wb'), protocol=4)



get_landsacpe_variable()
variable_name = 'lai'
mode='ssp585'
'''
read data
'''
file_name='lai_Lmon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc'
nc_obj=Dataset(fr'CMIP6_origin_data/{variable_name}/{file_name}')
variable_list,data=read_nc_aphro(nc_obj,print_info=True)
pickle.dump(data, open(fr'CMIP6_data/' + variable_name + '/'+file_name[0:-4]+'.pkl', 'wb'), protocol=4)
data=pickle.load(open(fr'CMIP6_data/' + variable_name + '/'+file_name[0:-4]+'.pkl', 'rb'))
data=get_year_data(data,variable_name)
variable_save,grid_save=match_coord(data,variable_name,mode)

