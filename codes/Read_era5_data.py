import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import ma
import pickle
import datetime,time
import math
import os

def read_nc_aphro(nc_obj,print_info=False,save=False):
    if print_info==True:
        #查看nc文件有些啥东东
        print(nc_obj)
        print('---------------------------------------')

        #查看nc文件中的变量
        print(nc_obj.variables.keys())
        for i in nc_obj.variables.keys():
            print(i)
        print('---------------------------------------')
    t=nc_obj.variables['time'][:]
    st = datetime.datetime(1900, 1, 1, 0, 0)
    a = st + datetime.timedelta(hours=int(t[-1]))
    print(a)
    if save==True:
        data={}
        variable_list=[]

        for i in tqdm(nc_obj.variables.keys()):
            variable_list.append(i)
            pickle.dump(nc_obj.variables[i][:], open(fr'era5_data/era5_{i}.pkl', 'wb'), protocol=4)
        return variable_list,data

def find_nearest(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return idx

def get_year_data(data,lat,lon):
    lat_loc=np.where((lat <= 54.875) & (lat >= 9.125))[0]
    lon_loc=np.where((lon <= 140.875) & (lon > 69.125))[0]

    #1950~2023
    data_month=data
    year_num=math.floor(data_month.shape[0]/12)
    for i in range(year_num):
        loc=ma.mean(data_month[12*i:12*(i+1)],axis=0).reshape(1,data_month.shape[1],data_month.shape[2])
        if i==0:
            data_year=loc
        else:
            data_year=ma.concatenate((data_year,loc),axis=0)

    data=data_year[:,np.min(lat_loc):(np.max(lat_loc)+1),np.min(lon_loc):(np.max(lon_loc)+1)]
    data_lon=lon[np.min(lon_loc):(np.max(lon_loc)+1)]
    data_lat= lat[np.min(lat_loc):(np.max(lat_loc) + 1)]
    print(data.shape)
    # plt.imshow(data[0])
    # plt.colorbar()
    # plt.show()
    return data,data_lon,data_lat
def match_coord(data,lat_era,lon_era,variable_name):
    data_era,data_lon,data_lat=get_year_data(data,lat_era,lon_era)
    with open('result_save/coord_save.pkl', 'rb') as f:
        coord_save = pickle.load(f)
    Dataset_1951_2015 = pickle.load(open(f'Dataset/Dataset_1951.pkl', 'rb'))
    lat = Dataset_1951_2015['latitude']
    lon = Dataset_1951_2015['longitude']
    variable_save=[]
    grid_save = ma.ones([data_era.shape[0], lat.shape[0], lon.shape[0]]) * (-1e8)
    for coord in tqdm(coord_save):
        lat_idx=find_nearest(data_lat,lat[coord[0]])
        lon_idx=find_nearest(data_lon,lon[coord[1]])
        variable_save.append(data_era[:,lat_idx,lon_idx])
        grid_save[:,coord[0],coord[1]]=data_era[:,lat_idx,lon_idx]
    grid_save = ma.masked_less(grid_save, -1e7)
    variable_save=np.vstack(variable_save)
    pickle.dump(variable_save, open(fr'era5_processed_data/variable_save_{variable_name}.pkl', 'wb'), protocol=4)
    pickle.dump(grid_save, open(fr'era5_processed_data/grid_save_{variable_name}.pkl', 'wb'), protocol=4)
    return variable_save,grid_save


# nc_obj=Dataset(fr'era5_data/era5_data_3.nc')
# read_nc_aphro(nc_obj,print_info=True)
# raise OSError
def batch_save_variable():
    #variable_list=['d2m','i10fg','msl','si10','sp','t2m','u10','u10n','u100','v10','v10n','v100','cbh', 'e', 'hcc', 'lcc', 'mwd', 'mwp', 'mcc', 'pev', 'ro', 'swh', 'ssro', 'sro', 'tcc', 'tciw', 'tclw', 'tp', 'p80.162', 'p79.162', 'p90.162', 'p88.162', 'p91.162', 'p89.162']
    variable_list=['cape','cvh','lai_hv','lai_lv','cvl']
    #d2m= pickle.load(open(f'era5_data/era5_d2m.pkl', 'rb'))
    lat_era= pickle.load(open(f'era5_data/era5_latitude.pkl', 'rb'))
    lon_era= pickle.load(open(f'era5_data/era5_longitude.pkl', 'rb'))
    for variable_name in variable_list:
        # variable_name='sst'
        variable= pickle.load(open(f'era5_data/era5_{variable_name}.pkl', 'rb'))
        print(variable.shape)
        if variable.ndim==4:
            match_coord(variable[:, 0, :, :], lat_era, lon_era, variable_name)
        else:
            match_coord(variable, lat_era, lon_era, variable_name)

def get_lai():
    cvl_grid = pickle.load(open(f'era5_processed_data/grid_save_cvl.pkl', 'rb'))
    cvl_row = pickle.load(open(f'era5_processed_data/variable_save_cvl.pkl', 'rb'))
    cvh_grid = pickle.load(open(f'era5_processed_data/grid_save_cvh.pkl', 'rb'))
    cvh_row = pickle.load(open(f'era5_processed_data/variable_save_cvh.pkl', 'rb'))
    lai_hv_grid = pickle.load(open(f'era5_processed_data/grid_save_lai_hv.pkl', 'rb'))
    lai_hv_row = pickle.load(open(f'era5_processed_data/variable_save_lai_hv.pkl', 'rb'))
    lai_lv_grid = pickle.load(open(f'era5_processed_data/grid_save_lai_lv.pkl', 'rb'))
    lai_lv_row = pickle.load(open(f'era5_processed_data/variable_save_lai_lv.pkl', 'rb'))
    lai_grid=cvl_grid*lai_lv_grid+cvh_grid*lai_hv_grid
    lai_row=cvl_row*lai_lv_row+cvh_row*lai_hv_row
    pickle.dump(lai_row, open(fr'era5_processed_data/variable_save_lai.pkl', 'wb'), protocol=4)
    pickle.dump(lai_grid, open(fr'era5_processed_data/grid_save_lai.pkl', 'wb'), protocol=4)
