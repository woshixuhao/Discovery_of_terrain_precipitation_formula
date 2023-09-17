# -*- coding: UTF-8 -*-
'''
The GA-GWR for discovering governing equation bentween terrain and average annual precipitation
'''
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import heapq
import math
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from tqdm import tqdm
import time
import numba
from numba import jit
import warnings
import copy
from multiprocessing import Process,Queue
import scipy
warnings.filterwarnings("ignore")
np.random.seed(5250)
random.seed(1998)
aspect_list=['#E9F2E9','#284757','#E9F2E9']
aspect_cmp = LinearSegmentedColormap.from_list('chaos',aspect_list)
slope_list=['#E9F2E9','#1C3B47']
slope_cmp = LinearSegmentedColormap.from_list('chaos',slope_list)
precip_list=['#E8F1F2','#718FC9','#7160A8','#6D3588','#531C46']
precip_cmp = LinearSegmentedColormap.from_list('chaos',precip_list)
def Dict_to_list(Dict):
    list = []
    for i in Dict:
        list.append(i)
    return list
def FiniteDiff_x(u, lat,dx, d):
    # 用二阶微分计算d阶微分，不过在三阶以上准确性会比较低
    # u是需要被微分的数据
    # dx是网格的空间大小
    nt, nx = u.shape
    ux = np.zeros([nt, nx])
    if d == 1:
        for i in range(nt):
            ux[i, 1:nx - 1] = (u[i, 2:nx] - u[i, 0:nx - 2]) / (2 * dx*np.cos(lat[i]/180*math.pi))
            ux[i, 0] = (-3.0 / 2 * u[i, 0] + 2 * u[i, 1] - u[i, 2] / 2) / (dx*np.cos(lat[i]/180*math.pi))
            ux[i, nx - 1] = (2.0 / 2 * u[i, nx - 1] - 2 * u[i, nx - 2] + u[i, nx - 3] / 2) / (dx*np.cos(lat[i]/180*math.pi))
        return ux
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
def show_dataset():
    plt.figure(1, figsize=(3, 3), dpi=300)
    plt.imshow(np.flip(mean_precio, axis=0))
    plt.colorbar(fraction=0.03)
    plt.tight_layout()
    plt.savefig('fig_save/mean_precip.tiff')
    plt.show()

    plt.figure(2, figsize=(3, 3), dpi=300)
    plt.imshow(np.flip(HT, axis=0))
    plt.colorbar(fraction=0.03)
    plt.tight_layout()
    # plt.savefig('fig_save/HT.tiff')
    plt.show()

    plt.figure(3, figsize=(3, 3), dpi=300)
    plt.imshow(np.flip(aspect, axis=0))
    plt.colorbar(fraction=0.03)
    plt.tight_layout()
    # plt.savefig('fig_save/aspect.tiff')
    plt.show()

    plt.figure(1, figsize=(3, 3), dpi=300)
    plt.imshow(np.flip(relief, axis=0))
    plt.colorbar(fraction=0.03)
    plt.tight_layout()
    # plt.savefig('fig_save/relief.tiff')
    plt.show()-0
def plot(target, ref,name='default', min='no', max='no', is_ref=False):
    plt.rcParams['font.family'] = 'Arial'
    if min == 'no':
        min = np.min(target)
        max = np.max(target)
    if is_ref == True:
        plt.figure(10, dpi=300)
        plt.subplot(2, 1, 1)
        plt.imshow(np.flip(target, axis=0), vmin=min, vmax=max)
        plt.colorbar(fraction=0.03)
        plt.subplot(2, 1, 2)
        plt.imshow(np.flip(ref, axis=0), vmin=min, vmax=max)
        plt.colorbar(fraction=0.03)
        # plt.savefig('fig_save/mean_precip.tiff')
        plt.show()
    else:

        plt.figure(10, dpi=300)
        plt.imshow(np.flip(target, axis=0), vmin=min, vmax=max, cmap=precip_cmp)
        plt.colorbar(fraction=0.03)
        plt.axis('off')
        plt.savefig(f'fig_save/{name}.png',dpi=300)
        plt.show()
def plot_error(target,name='default', min='no', max='no'):
    plt.rcParams['font.family'] = 'Arial'
    if min == 'no':
        min = np.min(target)
        max = np.max(target)

    plt.figure(10, dpi=300)
    plt.imshow(np.flip(target, axis=0), vmin=min, vmax=max, cmap='PuRd')  #PuRd  #coolwarm  #tab20c
    plt.colorbar(fraction=0.03)
    plt.axis('off')
    plt.savefig(f'fig_save/{name}.png',dpi=300)
    plt.show()
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

def combine_slice(lat,lon):
    #============Generate all slice==============
    print('============Generate all slice================\n')
    all_slice = []
    all_slice_mask = []
    lat_s = np.arange(wl, lat - wl, 1)
    lon_s = np.arange(wl, lon - wl, 1)
    result_save = []
    coef_save = []
    fitness_save = []
    power_save = []
    coord_save=[]
    for lat in tqdm(lat_s):
        for lon in lon_s:
            domain_precio = mean_precio[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1,:]
            domain_HT = HT[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_aspect = aspect[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_relief = relief[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_slope = slope[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_dHTdx = dHTdx[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_dHTdy = dHTdy[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_X = ma.squeeze(ma.array([
                          domain_HT.reshape(-1, 1),                  #0
                          domain_aspect.reshape(-1, 1),              #1
                          domain_relief.reshape(-1, 1),              #2
                          domain_slope.reshape(-1, 1),               #3
                          domain_dHTdx.reshape(-1,1),                #4
                          domain_dHTdy.reshape(-1,1),                #5
                          ma.sqrt(domain_HT.reshape(-1, 1)),         #6
                          ma.sin(domain_aspect).reshape(-1, 1),      #7
                          ma.cos(domain_aspect).reshape(-1, 1),      #8
                          ma.sin(domain_slope).reshape(-1, 1),       #9
                          ma.cos(domain_slope).reshape(-1,1)         #10
                          ])).T
            domain_y=domain_precio.reshape(domain_precio.shape[0]*domain_precio.shape[1], domain_precio.shape[2])
            domain_mask=ma.concatenate((domain_y,domain_X),axis=1)
            domain_no_mask = ma.compress_rows(domain_mask)
            central_point = ma.compress_rows(
                domain_mask[int((2 * wl + 1) * (2 * wl + 1) / 2), :].reshape(-1, domain_mask.shape[1]))

            if domain_no_mask.shape[0] > 5 and central_point.shape[0] != 0:
                all_slice.append(domain_no_mask)
                all_slice_mask.append(domain_mask)
                coord_save.append([lat,lon])
    with open('../result_save_mean_year/all_slice.pkl', 'wb') as f:
        pickle.dump(all_slice, f)
    with open('../result_save_mean_year/all_slice_mask.pkl', 'wb') as f:
        pickle.dump(all_slice_mask, f)
    with open('../result_save_mean_year/coord_save.pkl', 'wb') as f:
        pickle.dump(coord_save,f)


    print(f'length of all_slice is:{len(all_slice)}')
    print(f'length of all_slice_mask is:{len(all_slice_mask)}')
    #============Generate concat slice and split index
    with open('../result_save_mean_year/all_slice.pkl', 'rb') as f:
        all_slice= pickle.load(f)
    with open('../result_save_mean_year/all_slice_mask.pkl', 'rb') as f:
        all_slice_mask = pickle.load(f)
    index=0
    index_mask=0
    index_record=[]
    index_mask_record=[]
    print('============Concat all slice================\n')
    for i in tqdm(range(len(all_slice))):
        index_record.append(index)
        index+=all_slice[i].shape[0]
    index_record.append(index)
    concat_all = np.vstack(all_slice)
    with open('../result_save_mean_year/concat_all_slice.pkl', 'wb') as f:
        pickle.dump(concat_all,f)
    with open('../result_save_mean_year/split_index.pkl', 'wb') as f:
        pickle.dump(index_record,f)

    for i in tqdm(range(len(all_slice_mask))):
        index_mask_record.append(index_mask)
        index_mask += all_slice_mask[i].shape[0]
    index_mask_record.append(index_mask)
    concat_all_mask = ma.vstack(all_slice_mask)
    with open('../result_save_mean_year/concat_all_slice_mask.pkl', 'wb') as f:
        pickle.dump(concat_all_mask,f)
    with open('../result_save_mean_year/split_index_mask.pkl', 'wb') as f:
        pickle.dump(index_mask_record,f)
    print(concat_all.shape)
    print(concat_all_mask.shape)


    print(f'length of concat_all_slice is:{len(concat_all)}')
    print(f'length of concat_all_slice_mask is:{len(concat_all)}')

def combine_slice_validate(lat,lon,large_precio,small_precio,large_HT,small_HT,large_relief,small_relief):
    #=========read_data=================
    years = np.arange(2011, 2015, 1)
    all_year_precip = 0
    for year in tqdm(years):
        Dataset_year = pickle.load(open(f'Dataset/Dataset_{year}.pkl', 'rb'))
        precip_year = ma.sum(Dataset_year['precip'], axis=0).reshape(-1, 1)
        if year == 2011:
            all_year_precip = precip_year
        else:
            all_year_precip = ma.concatenate((all_year_precip, precip_year), axis=1)
    mean_precio = all_year_precip.reshape(184, 288, years.shape[0])

    Dataset_1951_2015 = pickle.load(open('../Dataset_1951_2015.pkl', 'rb'))
    HT = Dataset_1951_2015['HT'][0]  # shape(184,288)
    aspect = Dataset_1951_2015['spect']
    relief = Dataset_1951_2015['relief']
    slope = Dataset_1951_2015['slope']
    lat = Dataset_1951_2015['latitude']
    lon = Dataset_1951_2015['longitude']
    lat_num = lat.shape[0]  # lat = 184
    lon_num = lon.shape[0]  # lon = 288

    wl = 3  # window length
    # ========Standardlization==================
    HT =  standardlization(HT)
    aspect = standardlization_aspect(aspect)
    relief = standardlization(relief)
    mean_precio = (mean_precio - small_precio) / (large_precio - small_precio)
    slope = slope / 180 * math.pi
    dHTdx = FiniteDiff_x(HT, 1, d=1)
    dHTdy = FiniteDiff_y(HT.T, 1, d=1).T
    # =========lat and lon==========
    domain_lat = ma.zeros([lat_num, lon_num])
    domain_lon = ma.zeros([lat_num, lon_num])
    for i in range(lat_num):
        domain_lat[i] = lat[i]
    for i in range(lon_num):
        domain_lon[:, i] = lon[i]

    lat=lat_num
    lon=lon_num

    #============Generate all slice==============
    print('============Generate all slice================\n')
    all_slice = []
    all_slice_mask = []
    lat_s = np.arange(wl, lat - wl, 1)
    lon_s = np.arange(wl, lon - wl, 1)
    result_save = []
    coef_save = []
    fitness_save = []
    power_save = []
    coord_save=[]
    for lat in tqdm(lat_s):
        for lon in lon_s:
            domain_precio = mean_precio[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1,:]
            domain_HT = HT[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_aspect = aspect[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_relief = relief[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_slope = slope[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_dHTdx = dHTdx[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_dHTdy = dHTdy[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
            domain_X = ma.squeeze(ma.array([
                          domain_HT.reshape(-1, 1),                  #0
                          domain_aspect.reshape(-1, 1),              #1
                          domain_relief.reshape(-1, 1),              #2
                          domain_slope.reshape(-1, 1),               #3
                          domain_dHTdx.reshape(-1,1),                #4
                          domain_dHTdy.reshape(-1,1),                #5
                          ma.sqrt(domain_HT.reshape(-1, 1)),         #6
                          ma.sin(domain_aspect).reshape(-1, 1),      #7
                          ma.cos(domain_aspect).reshape(-1, 1),      #8
                          ma.sin(domain_slope).reshape(-1, 1),       #9
                          ma.cos(domain_slope).reshape(-1,1)         #10
                          ])).T
            domain_y=domain_precio.reshape(domain_precio.shape[0]*domain_precio.shape[1], domain_precio.shape[2])
            domain_mask=ma.concatenate((domain_y,domain_X),axis=1)
            domain_no_mask = ma.compress_rows(domain_mask)
            central_point = ma.compress_rows(
                domain_mask[int((2 * wl + 1) * (2 * wl + 1) / 2), :].reshape(-1, domain_mask.shape[1]))

            if domain_no_mask.shape[0] > 5 and central_point.shape[0] != 0:
                all_slice.append(domain_no_mask)
                all_slice_mask.append(domain_mask)
                coord_save.append([lat,lon])
    with open('../result_save/all_slice_valiate.pkl', 'wb') as f:
        pickle.dump(all_slice, f)
    with open('../result_save/all_slice_mask_valiate.pkl', 'wb') as f:
        pickle.dump(all_slice_mask, f)
    with open('../result_save/coord_save_valiate.pkl', 'wb') as f:
        pickle.dump(coord_save,f)


    print(f'length of all_slice is:{len(all_slice)}')
    print(f'length of all_slice_mask is:{len(all_slice_mask)}')
    #============Generate concat slice and split index
    with open('../result_save/all_slice_valiate.pkl', 'rb') as f:
        all_slice= pickle.load(f)
    with open('../result_save/all_slice_mask_valiate.pkl', 'rb') as f:
        all_slice_mask = pickle.load(f)
    index=0
    index_mask=0
    index_record=[]
    index_mask_record=[]
    print('============Concat all slice================\n')
    for i in tqdm(range(len(all_slice))):
        index_record.append(index)
        index+=all_slice[i].shape[0]
    index_record.append(index)
    concat_all = np.vstack(all_slice)
    with open('../result_save/concat_all_slice_valiate.pkl', 'wb') as f:
        pickle.dump(concat_all,f)
    with open('../result_save/split_index_valiate.pkl', 'wb') as f:
        pickle.dump(index_record,f)

    for i in tqdm(range(len(all_slice_mask))):
        index_mask_record.append(index_mask)
        index_mask += all_slice_mask[i].shape[0]
    index_mask_record.append(index_mask)
    concat_all_mask = ma.vstack(all_slice_mask)
    with open('../result_save/concat_all_slice_mask_valiate.pkl', 'wb') as f:
        pickle.dump(concat_all_mask,f)
    with open('../result_save/split_index_mask_valiate.pkl', 'wb') as f:
        pickle.dump(index_mask_record,f)
    print(concat_all.shape)
    print(concat_all_mask.shape)


    print(f'length of concat_all_slice is:{len(concat_all)}')
    print(f'length of concat_all_slice_mask is:{len(concat_all)}')
class GA():
    def __init__(self, concat_all_slice,concat_all_slice_mask,split_index,split_index_mask,year_shape,wl):
        self.max_length = 7
        self.max_multiple_length=3
        self.partial_prob = 0.6
        self.genes_prob = 0.6
        self.mutate_rate = 0.4
        self.delete_rate = 0.5
        self.add_rate = 0.4
        self.pop_size = 200
        self.n_generations = 200
        self.epi =1e-5
        self.year_shape = year_shape
        self.wl = wl
        self.concat_X_slice=concat_all_slice
        self.concat_X_slice_mask = concat_all_slice_mask
        self.split_index=split_index
        self.split_index_mask=split_index_mask
        self.X_num=concat_all_slice.shape[0]
        self.X_mask_num = concat_all_slice_mask.shape[0]
        self.var_num=self.concat_X_slice.shape[1]-self.year_shape
        self.add_constant=True
        self.length_calculation='Single' #['Single','All']


    def random_module(self):
        genes_module = []
        for i in range(self.max_multiple_length):
            a = random.randint(0, self.var_num - 1)
            genes_module.append(a)
            prob = random.uniform(0, 1)
            if prob > self.partial_prob:
                break
        return genes_module

    def random_genome(self):
        genes = []
        for i in range(self.max_length):
            gene_random = GA.random_module(self)
            genes.append(sorted(gene_random))
            prob = random.uniform(0, 1)
            if prob > self.genes_prob:
                break
        return genes

    def translate_DNA(self, gene):
        gene_translate = np.ones([self.X_num, 1])
        gene_translate_mask = np.ones([self.X_mask_num, 1])
        length_penalty_coef = 0
        self.X = self.concat_X_slice[:, self.year_shape:]
        self.X_mask = self.concat_X_slice_mask[:, self.year_shape:]
        self.y = self.concat_X_slice[:, 0:self.year_shape]
        self.y_mask = self.concat_X_slice_mask[:, 0:self.year_shape]
        for k in range(len(gene)):
            gene_module = gene[k]
            if self.length_calculation == 'All':
                length_penalty_coef += len(gene_module)
            module_out = np.ones([self.X_num, 1])
            module_out_mask = np.ones([self.X_mask_num, 1])
            for i in gene_module:
                module_out *= self.X[:, i].reshape(-1, 1)
                module_out_mask *= self.X_mask[:, i].reshape(-1, 1)
            gene_translate = np.hstack((gene_translate, module_out))
            gene_translate_mask = np.hstack((gene_translate_mask, module_out_mask))
        if self.add_constant==False:
            gene_translate = np.delete(gene_translate, [0], axis=1)
            gene_translate_mask = np.delete(gene_translate_mask, [0], axis=1)
        self.gene_translate=gene_translate
        self.gene_translate_mask=gene_translate_mask
        if self.length_calculation=='Single':
            self.length_penalty_coef = len(gene)
        elif self.length_calculation=='All':
            self.length_penalty_coef=length_penalty_coef
        return self.gene_translate,self.gene_translate_mask,self.y,self.y_mask,self.length_penalty_coef

    def get_fitness(self):
        self.all_pred = []
        self.all_true = []
        # time_1=time.time()
        for iter in range(len(self.split_index) - 1):
            lst = np.linalg.lstsq(self.gene_translate[self.split_index[iter]:self.split_index[iter + 1]],
                                  self.y[self.split_index[iter]:self.split_index[iter + 1]],
                                  rcond=None)
            coef = lst[0]
            target_mask = np.dot(
                self.gene_translate_mask[self.split_index_mask[iter]:self.split_index_mask[iter + 1]], coef)[
                          int((2 * self.wl + 1) * (2 * self.wl + 1) / 2), :]
            self.all_pred.append(target_mask)
            self.all_true.append(self.y_mask[self.split_index_mask[iter]:self.split_index_mask[iter + 1]][
                                     int((2 * self.wl + 1) * (2 * self.wl + 1) / 2)])
        # time_2 = time.time()
        # print(time_2-time_1)
        y = np.vstack(self.all_pred)
        target = np.vstack(self.all_true)
        MSE = np.mean((y - target) ** 2)
        R_2 = (1 - (((y - target) ** 2).sum() / ((target - target.mean()) ** 2).sum()))
        MSE_total = np.mean((y - target) ** 2) + self.epi * self.length_penalty_coef

        return MSE, R_2, MSE_total

    def cross_over(self):
        Chrom, size_pop = self.Chrom, self.pop_size
        Chrom1, Chrom2 = Chrom[::2], Chrom[1::2]
        random.shuffle(Chrom1)
        random.shuffle(Chrom2)
        for i in range(int(size_pop / 2)):
            n1 = np.random.randint(0, len(Chrom1[i]))
            n2 = np.random.randint(0, len(Chrom2[i]))

            father = copy.deepcopy(Chrom1[i][n1])
            mother = copy.deepcopy(Chrom2[i][n2])

            Chrom1[i][n1] = mother
            Chrom2[i][n2] = father

        Chrom[::2], Chrom[1::2] = Chrom1, Chrom2
        self.Chrom = Chrom
        return self.Chrom

    def mutation(self):
        Chrom, size_pop = self.Chrom, self.pop_size

        for i in range(size_pop):
            n1 = np.random.randint(0, len(Chrom[i]))

            # ------------add module---------------
            prob = np.random.uniform(0, 1)
            if prob < self.add_rate:
                add_Chrom = GA.random_module(self)
                if add_Chrom not in Chrom[i]:
                    Chrom[i].append(add_Chrom)

            # --------delete module----------------
            prob = np.random.uniform(0, 1)
            if prob < self.delete_rate:
                if len(Chrom[i]) > 1:
                    delete_index = np.random.randint(0, len(Chrom[i]))
                    Chrom[i].pop(delete_index)
            # ------------gene mutation------------------
            prob = np.random.uniform(0, 1)
            if prob < self.mutate_rate:
                n1 = np.random.randint(0, len(Chrom[i]))
                n2 = np.random.randint(0, len(Chrom[i][n1]))
                Chrom[i][n1][n2] = random.randint(0, self.var_num - 1)

        self.Chrom = Chrom
        return self.Chrom

    def multi_process(self, Chrom, que,que_R,que_chrom):
        fitness_list = []
        R_2_list = []
        Chrom_list=[]
        for i in range(len(Chrom)):
            gene_translate, gene_translate_mask, y, y_mask, length_penalty_coef = GA.translate_DNA(self, Chrom[i])
            MSE, R_2, MSE_total = GA.get_fitness(self)
            fitness_list.append(MSE_total)
            R_2_list.append(R_2)
            Chrom_list.append(Chrom[i])

        que.put(fitness_list)
        que_R.put(R_2_list)
        que_chrom.put(Chrom_list)

    def select(self):  # nature selection wrt pop's fitness
        Chrom, size_pop = self.Chrom, self.pop_size
        new_Chrom = []
        new_fitness = []
        new_R_2 = []
        Chrom_list=[]
        fitness_list = []
        R_2_list = []

        tube_num = 5
        process = list()
        que = Queue()
        que_R = Queue()
        que_C = Queue()
        total = []
        dx = int(len(Chrom) / tube_num)
        for j in range(tube_num):
            p = Process(target=GA.multi_process, args=(self, Chrom[j * dx:(j + 1) * dx], que,que_R,que_C))
            process.append(p)
            p.start()
        start_time = time.time()

        for th_item in process:
            while th_item.is_alive():
                while False == que.empty():
                    fitness_list.extend(que.get())
                    R_2_list.extend(que_R.get())
                    Chrom_list.extend(que_C.get())
        for p in process:
            p.join()
        re1 = list(map(fitness_list.index, heapq.nsmallest(int(size_pop / 2), fitness_list)))

        for index in re1:
            new_Chrom.append(Chrom_list[index])
            new_fitness.append(fitness_list[index])
            new_R_2.append(R_2_list[index])
        for index in range(int(size_pop / 2)):
            new = GA.random_genome(self)
            new_Chrom.append(new)

        self.Chrom = new_Chrom
        self.Fitness = new_fitness
        self.R_2 =new_R_2
        return self.Chrom, self.Fitness, self.R_2


    def delete_duplicates(self):
        Chrom, size_pop = self.Chrom, self.pop_size
        for i in range(size_pop):
            new_genome = []
            for j in range(len(Chrom[i])):
                if sorted(Chrom[i][j]) not in new_genome:
                    new_genome.append(sorted(Chrom[i][j]))
            Chrom[i] = new_genome
        self.Chrom = Chrom
        return self.Chrom

    def random_on_basis(self,best_chrom):
        change_mode=np.random.randint(0,4)
        if change_mode==0:
            n1 = np.random.randint(0, len(best_chrom))
            new_chrom=copy.deepcopy(best_chrom)
            new_chrom[n1]=GA.random_module(self)
        if change_mode==1:
            n1 = np.random.randint(0, len(best_chrom))
            new_chrom=GA.random_genome(self)
            new_chrom.pop(0)
            new_chrom.append(best_chrom[n1])
        if change_mode==2:
            n1 = np.random.randint(0, len(best_chrom))
            new_chrom = copy.deepcopy(best_chrom)
            # ------------add module---------------
            prob = np.random.uniform(0, 1)
            if prob < self.add_rate:
                add_Chrom = GA.random_module(self)
                if add_Chrom not in new_chrom:
                    new_chrom.append(add_Chrom)

            # --------delete module----------------
            prob = np.random.uniform(0, 1)
            if prob < self.delete_rate:
                if len(new_chrom) > 1:
                    delete_index = np.random.randint(0, len(new_chrom))
                    new_chrom.pop(delete_index)
            # ------------gene mutation------------------
            prob = np.random.uniform(0, 1)
            if prob < self.mutate_rate:
                n1 = np.random.randint(0, len(new_chrom))
                n2 = np.random.randint(0, len(new_chrom[n1]))
                new_chrom[n1][n2] = random.randint(0, self.var_num - 1)
        if change_mode==3:
            new_chrom=GA.random_genome(self)
        return new_chrom


    def distinction(self):
        Chrom=copy.deepcopy(self.Chrom)
        for m in range(3,len(Chrom)):
            Chrom[m]=GA.random_on_basis(self,self.Chrom[0])
        self.Chrom=Chrom

    def evolution(self, print_info=False):
        self.Chrom = []
        self.Fitness = []
        self.R_2=[]
        print('----------initial-------------')
        for iter in tqdm(range(self.pop_size)):
            intial_genome = GA.random_genome(self)
            self.Chrom.append(intial_genome)

        GA.delete_duplicates(self)
        GA.select(self)
        print(self.Chrom[0:5])
        print(self.R_2[0:5])
        print(self.Fitness[0:5])


        distinction_time=0
        for iter in tqdm(range(self.n_generations)):
            with open('../../best_save.pkl', 'wb') as f:
                pickle.dump(copy.deepcopy(self.Chrom)[0], f)
            best = copy.deepcopy(self.Chrom)[0]
            GA.cross_over(self)
            GA.mutation(self)
            GA.delete_duplicates(self)
            with open('../../best_save.pkl', 'rb') as f:
                best = pickle.load(f)
            self.Chrom[0] = best
            GA.select(self)
            if print_info == True:
                if self.Chrom[0]!=best:
                    print(f'The best Chrom: {self.Chrom[0]}')
                    print(f'The best fitness: {self.Fitness[0]}')
                    print(f'The best R_2: {self.R_2[0]}')
                else:
                    distinction_time+=1
                    #print(f"distinction_time is:{distinction_time}")
            if distinction_time==4:
                #print('It is distiction!')
                # print('before distinction:',self.Chrom[0:6])
                GA.distinction(self)
                #print('after distinction:', self.Chrom[0:6])
                distinction_time=0
        return self.Chrom[0],self.Fitness[0],self.R_2[0]

if __name__ == '__main__':
    years=np.arange(1951,2016,1)
    all_year_precip = 0
    for year in years:
        Dataset_year = pickle.load(open(f'Dataset/Dataset_{year}.pkl', 'rb'))
        precip_year = Dataset_year['precip']
        all_year_precip += ma.sum(precip_year, axis=0)
    mean_precio = all_year_precip / years.shape[0]
    mean_precio=mean_precio.reshape(mean_precio.shape[0],mean_precio.shape[1],1)
    Dataset_1951_2015 = pickle.load(open(f'../Dataset/Dataset_1951.pkl', 'rb'))
    HT = Dataset_1951_2015['HT'][0]  # shape(184,288)
    aspect = Dataset_1951_2015['spect']
    relief = Dataset_1951_2015['relief']
    slope = Dataset_1951_2015['slope']
    lat = Dataset_1951_2015['latitude']
    lon = Dataset_1951_2015['longitude']
    lat_num = lat.shape[0]  #lat = 184
    lon_num = lon.shape[0]  #lon = 288


    wl = 3 #window length
    large_precio=ma.max(mean_precio)
    small_precio=ma.min(mean_precio)
    large_HT=ma.max(HT)
    small_HT=ma.min(HT)
    large_relief=ma.max(relief)
    small_relief=ma.min(relief)
    # ========Standardlization==================
    HT = standardlization(HT)
    aspect = standardlization_aspect(aspect)
    relief = standardlization(relief)
    mean_precio = standardlization(mean_precio)
    slope = slope/180*math.pi
    dHTdx = FiniteDiff_x(HT, lat, 111 * 0.25, d=1)
    dHTdy = FiniteDiff_y(HT.T, 111 * 0.25, d=1).T
    # =========lat and lon==========
    domain_lat = ma.zeros([lat_num, lon_num])
    domain_lon = ma.zeros([lat_num, lon_num])
    for i in range(lat_num):
        domain_lat[i] = lat[i]
    for i in range(lon_num):
        domain_lon[:,i] = lon[i]


    # =============Split  network============
    MODEL='GA_GWR'
    if MODEL=='LSTSQ':
        domain_mask = ma.squeeze(
            ma.array([mean_precio.reshape(-1, 1),
                      HT.reshape(-1, 1),  # 0
                      #aspect.reshape(-1, 1),  # 1
                      #relief.reshape(-1, 1),  # 2
                      #slope.reshape(-1, 1),  # 3
                      domain_lat.reshape(-1,1),
                      domain_lon.reshape(-1,1),
                      # ma.ones([HT.shape[0],HT.shape[1]]).reshape(-1,1),
                      # dHTdx.reshape(-1, 1),  # 4
                      # dHTdy.reshape(-1, 1)*aspect.reshape(-1, 1),  # 5
                      # ma.sqrt(HT.reshape(-1, 1)),  # 6
                      ma.sin(aspect).reshape(-1, 1),  # 7
                      ma.cos(aspect).reshape(-1, 1),  # 8
                      # ma.sin(slope).reshape(-1, 1),  # 9
                      # ma.cos(slope).reshape(-1, 1)  # 10
                      ])).T

        domain_no_mask = ma.compress_rows(domain_mask)
        X=domain_no_mask[:,1:]
        y=domain_no_mask[:,0].reshape(-1,1)
        X_mask=domain_mask[:,1:]
        y_mask=domain_mask[:,0]
        coef = np.linalg.lstsq(X, y, rcond=None)[0].reshape(-1, 1)
        target = 0
        target_mask = 0

        for i in range(coef.shape[0]):
            target += coef[i, 0] * X[:, i]
            target_mask += coef[i, 0] * X_mask[:, i]
        target = target.reshape(-1, 1)
        target_mask = target_mask.reshape(-1, 1)

        y=y*(large_precio-small_precio)+small_precio
        target=target*(large_precio-small_precio)+small_precio
        print(np.sqrt(np.mean((y - target) ** 2)))
        print(1 - (((y - target) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        print(np.corrcoef(y.reshape(-1, ), target.reshape(-1, )))

        plot(target_mask.reshape(184,288)*(large_precio-small_precio)+small_precio,
             target_mask.reshape(184,288)*(large_precio-small_precio)+small_precio,
             name='pred_OLS_3')
    if MODEL == 'GWR':
        lat=184
        lon=288
        all_true = []
        all_predict = []
        lat_s = np.arange(wl, lat-wl,1)
        lon_s = np.arange(wl, lon-wl,1)
        result_save = []
        coef_save = []
        fitness_save = []
        power_save = []
        all_p = ma.ones([lat, lon]) * (-1e8)
        all_t = ma.ones([lat, lon]) * (-1e8)
        for lat in tqdm(lat_s):
            for lon in lon_s:
                domain_precio = mean_precio[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1, :]
                domain_HT = HT[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_aspect = aspect[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_relief = relief[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_slope = slope[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_dHTdx = dHTdx[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_dHTdy = dHTdy[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_lat_local=domain_lat[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_lon_local=domain_lon[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_X = ma.squeeze(
                    ma.array([domain_HT.reshape(-1, 1),  # 0
                              domain_aspect.reshape(-1, 1),  # 1
                              domain_relief.reshape(-1, 1),  # 2
                              domain_slope.reshape(-1, 1),  # 3
                              # domain_dHTdx.reshape(-1, 1),  # 4
                              # domain_dHTdy.reshape(-1, 1),  # 5
                              # ma.sqrt(domain_HT.reshape(-1, 1)),  # 6
                              # ma.sin(domain_aspect).reshape(-1, 1),  # 7
                              # ma.cos(domain_aspect).reshape(-1, 1),  # 8
                              # ma.sin(domain_slope).reshape(-1, 1),  # 9
                              # ma.cos(domain_slope).reshape(-1, 1),  # 10
                              ma.ones([domain_HT.shape[0], domain_HT.shape[1]]).reshape(-1, 1),
                              # domain_lat_local.reshape(-1, 1),
                              # domain_lon_local.reshape(-1,1)
                              ])).T

                domain_y = domain_precio.reshape(domain_precio.shape[0] * domain_precio.shape[1],
                                                 domain_precio.shape[2])
                domain_mask = ma.concatenate((domain_y, domain_X), axis=1)
                domain_no_mask = ma.compress_rows(domain_mask)
                central_point = ma.compress_rows(
                    domain_mask[int((2 * wl + 1) * (2 * wl + 1) / 2), :].reshape(-1, domain_mask.shape[1]))

                if domain_no_mask.shape[0] > 5 and central_point.shape[0]!=0:
                    y = domain_no_mask[:, 0:1]
                    X = domain_no_mask[:, 1:]
                    y_mask=domain_mask[:,0:1]
                    X_mask=domain_mask[:, 1:]
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    target = 0
                    target_mask=0
                    target=np.dot(X,coef)
                    target_mask=ma.dot(X_mask,coef)


                    all_true.extend(y_mask[int((2*wl+1)*(2*wl+1)/2),:].tolist())
                    all_predict.extend(target_mask[int((2*wl+1)*(2*wl+1)/2),:].tolist())
                    all_p[lat, lon] =target_mask[int((2*wl+1)*(2*wl+1)/2),:]
                    all_t[lat, lon] = y_mask[int((2*wl+1)*(2*wl+1)/2),:]


                    #print(target_mask[int((2*wl+1)*(2*wl+1)/2)])

                else:
                    result_save.append([])
                    coef_save.append([])
                    fitness_save.append([])
                    power_save.append([])
        # np.save('result_save/all_slice.npy', all_slice)
        # np.save('result_save/all_slice_mask.npy', all_slice_mask)
        all_true = np.array(all_true)
        print(all_true.shape)
        all_predict = np.array(all_predict)
        y = all_true*(large_precio-small_precio)+small_precio
        target = all_predict*(large_precio-small_precio)+small_precio

        print(np.sqrt(np.mean((y-target)**2)))
        print(1 - (((y - target) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        print(np.corrcoef(y.reshape(-1,),target.reshape(-1,)))


        y = all_t*(large_precio-small_precio)+small_precio
        target = all_p*(large_precio-small_precio)+small_precio
        y= ma.masked_less(y,-1e7)
        target = ma.masked_less(target, -1e7)
        relative_error=ma.abs((y-target)/y)
        #plot_error(relative_error,'relative_year_2',min=0,max=1.75)

    if MODEL == 'GWR_validate':
        years = np.arange(2011, 2015, 1)
        all_year_precip = 0
        for year in tqdm(years):
            Dataset_year = pickle.load(open(f'Dataset/Dataset_{year}.pkl', 'rb'))
            precip_year = ma.sum(Dataset_year['precip'], axis=0).reshape(-1, 1)
            if year == 2011:
                all_year_precip = precip_year
            else:
                all_year_precip = ma.concatenate((all_year_precip, precip_year), axis=1)
        mean_precio = all_year_precip.reshape(184, 288, years.shape[0])
        mean_precio=(mean_precio-small_precio)/(large_precio-small_precio)
        lat=184
        lon=288
        all_true = []
        all_predict = []
        lat_s = np.arange(wl, lat-wl,1)
        lon_s = np.arange(wl, lon-wl,1)
        result_save = []
        coef_save = []
        fitness_save = []
        power_save = []
        all_p = ma.ones([years.shape[0],lat, lon]) * (-1e8)
        all_t = ma.ones([years.shape[0],lat, lon]) * (-1e8)
        for lat in tqdm(lat_s):
            for lon in lon_s:
                domain_precio = mean_precio[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1, :]
                domain_HT = HT[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_aspect = aspect[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_relief = relief[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_slope = slope[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_dHTdx = dHTdx[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_dHTdy = dHTdy[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_lat_local=domain_lat[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_lon_local=domain_lon[lat - wl:lat + wl + 1, lon - wl:lon + wl + 1]
                domain_X = ma.squeeze(
                    ma.array([domain_HT.reshape(-1, 1),  # 0
                              #domain_aspect.reshape(-1, 1),  # 1
                              #domain_relief.reshape(-1, 1),  # 2
                              #domain_slope.reshape(-1, 1),  # 3
                              # domain_dHTdx.reshape(-1, 1),  # 4
                              # domain_dHTdy.reshape(-1, 1),  # 5
                              # ma.sqrt(domain_HT.reshape(-1, 1)),  # 6
                              ma.sin(domain_aspect).reshape(-1, 1),  # 7
                              ma.cos(domain_aspect).reshape(-1, 1),  # 8
                              # ma.sin(domain_slope).reshape(-1, 1),  # 9
                              # ma.cos(domain_slope).reshape(-1, 1),  # 10
                              #ma.ones([domain_HT.shape[0], domain_HT.shape[1]]).reshape(-1, 1),
                              domain_lat_local.reshape(-1, 1),
                              domain_lon_local.reshape(-1,1)
                              ])).T

                domain_y = domain_precio.reshape(domain_precio.shape[0] * domain_precio.shape[1],
                                                 domain_precio.shape[2])
                domain_mask = ma.concatenate((domain_y, domain_X), axis=1)
                domain_no_mask = ma.compress_rows(domain_mask)
                central_point = ma.compress_rows(
                    domain_mask[int((2 * wl + 1) * (2 * wl + 1) / 2), :].reshape(-1, domain_mask.shape[1]))

                if domain_no_mask.shape[0] > 5 and central_point.shape[0]!=0:
                    y = domain_no_mask[:, 0:years.shape[0]]
                    X = domain_no_mask[:, years.shape[0]:]
                    y_mask=domain_mask[:,0:years.shape[0]]
                    X_mask=domain_mask[:, years.shape[0]:]
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    target = 0
                    target_mask=0
                    target=np.dot(X,coef)
                    target_mask=ma.dot(X_mask,coef)


                    all_true.extend(y_mask[int((2*wl+1)*(2*wl+1)/2),:].tolist())
                    all_predict.extend(target_mask[int((2*wl+1)*(2*wl+1)/2),:].tolist())
                    all_p[:,lat, lon] =target_mask[int((2*wl+1)*(2*wl+1)/2),:]
                    all_t[:,lat, lon] = y_mask[int((2*wl+1)*(2*wl+1)/2),:]


                    #print(target_mask[int((2*wl+1)*(2*wl+1)/2)])

                else:
                    result_save.append([])
                    coef_save.append([])
                    fitness_save.append([])
                    power_save.append([])
        # np.save('result_save/all_slice.npy', all_slice)
        # np.save('result_save/all_slice_mask.npy', all_slice_mask)
        all_true = np.array(all_true)
        print(all_true.shape)
        all_predict = np.array(all_predict)
        y = all_true*(large_precio-small_precio)+small_precio
        target = all_predict*(large_precio-small_precio)+small_precio

        print(np.sqrt(np.mean((y-target)**2)))
        print(1 - (((y - target) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        print(np.corrcoef(y.reshape(-1,),target.reshape(-1,)))


        y = all_t*(large_precio-small_precio)+small_precio
        target = all_p*(large_precio-small_precio)+small_precio
        y= ma.masked_less(y,-1e7)
        target = ma.masked_less(target, -1e7)
        relative_error=ma.abs((y-target)/y)
        plot_error(relative_error,'relative_year_2',min=0,max=1.75)
    if MODEL == 'GA_GWR':
        combine_slice(lat_num,lon_num)
        with open('../result_save_mean_year/all_slice.pkl', 'rb') as f:
            all_slice= pickle.load(f)
        with open('../result_save_mean_year/all_slice_mask.pkl', 'rb') as f:
            all_slice_mask = pickle.load(f)
        with open('../result_save_mean_year/concat_all_slice.pkl', 'rb') as f:
            concat_all_slice = pickle.load(f)
        with open('../result_save_mean_year/split_index.pkl', 'rb') as f:
            split_index = pickle.load(f)
        with open('../result_save_mean_year/concat_all_slice_mask.pkl', 'rb') as f:
            concat_all_slice_mask = pickle.load(f)
        with open('../result_save_mean_year/split_index_mask.pkl', 'rb') as f:
            split_index_mask = pickle.load(f)


        print('=============Start GA====================\n')
        year_shape=1
        optimizer = GA(concat_all_slice,concat_all_slice_mask,split_index,split_index_mask,year_shape,wl)
        print(f'The size of candidate library is: {concat_all_slice.shape[1]-year_shape}')
        print(f'The epi for GA is {optimizer.epi}')

        optimizer.evolution(print_info=True)


