from h5py import File
import numpy as np
from tqdm import tqdm
from numpy import ma
import pickle
years=np.arange(2008,2016,1)
TOPO_name=['aspect','height','relief_r100km','slope','subrelief']
def read_h5py_aphro(nc_obj):
    '''
    2008年后的数据是h5py格式，需要用此函数读取
    '''


    # 获取所有变量列表
    all_vars = list(nc_obj.keys())

    lat = nc_obj[all_vars[0]][:]
    lon=nc_obj[all_vars[1]][:]
    precip=nc_obj[all_vars[2]][:]
    rstn=nc_obj[all_vars[3]][:]
    time=nc_obj[all_vars[4]][:]
    return [time,lon,lat,precip,rstn]

def save_2008_2015():
    for i in tqdm(range(years.shape[0])):
        year=years[i]
        filename = fr'D:\张东晓研究-服务器\陈云天可解释\Prec.vs.TOPO-para\Prec.vs.TOPO-para\APHRO_MA_025deg_V1101.{year}.nc'
        nc_aphro = File(filename, mode='r')
        time=read_h5py_aphro(nc_aphro)[0]
        longitude=read_h5py_aphro(nc_aphro)[1]
        latitude=read_h5py_aphro(nc_aphro)[2]

        if i==0:
            precip = read_h5py_aphro(nc_aphro)[3]
            rstn = read_h5py_aphro(nc_aphro)[4]
        else:
            precip=ma.concatenate((precip,read_h5py_aphro(nc_aphro)[3]),axis=0)
            rstn=ma.concatenate((rstn,read_h5py_aphro(nc_aphro)[4]),axis=0)

    precip= ma.masked_less(precip,0)
    Dataset={'precip':precip,'rstn':rstn,'longitude':longitude,'latitude':latitude}
    pickle.dump(Dataset, open('Dataset_2008_2015.pkl', 'wb'), protocol=4)

def save_year():
    for i in tqdm(range(years.shape[0])):
        year=years[i]
        filename = fr'D:\张东晓研究-服务器\陈云天可解释\Prec.vs.TOPO-para\Prec.vs.TOPO-para\APHRO_MA_025deg_V1101.{year}.nc'
        nc_aphro = File(filename, mode='r')
        time=read_h5py_aphro(nc_aphro)[0]
        longitude=read_h5py_aphro(nc_aphro)[1]
        latitude=read_h5py_aphro(nc_aphro)[2]

        precip = read_h5py_aphro(nc_aphro)[3]
        rstn = read_h5py_aphro(nc_aphro)[4]

        precip= ma.masked_less(precip,0)
        Dataset={'precip':precip,'rstn':rstn,'longitude':longitude,'latitude':latitude}
        pickle.dump(Dataset, open(f'Dataset/Dataset_{year}.pkl', 'wb'), protocol=4)

