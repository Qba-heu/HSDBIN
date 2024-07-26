# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers of DeepHyperX (https://github.com/deephyper/deephyper).
"""
import h5py
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import os
from tqdm import tqdm
import seaborn as sns

import sklearn.model_selection
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from scipy import io
from sklearn.decomposition import PCA
import imageio
import itertools,visdom
import scipy.io as sio
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve



DATASETS_CONFIG = {
        'PaviaC': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
                     'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'PaviaU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'KSC': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                     'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                     'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'Indian_pines_corrected.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'Botswana': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                     'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
            'img': 'Botswana.mat',
            'gt': 'Botswana_gt.mat',
            },
        'houston2013': {
            'urls': [#'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                      #'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'Houston2013.mat',
            'gt': 'Houston2013_gt.mat',
            },
        'houston2018': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
            # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
            ],
            'img': 'Houston2013.mat',
            'gt': 'Houston2013_gt.mat',
        },
        'YRE': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'Huanghe_obt': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'YC': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'XiongAn': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },

        'ShangHai': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            },
        'HangZhou': {
            'urls': [  # 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                        # 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
                    ],
            'img': 'HuangHe.mat',
            'gt': 'HuangHe_gt.mat',
            }

    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def apply_pca(X,numComp=75):
    newX=np.reshape(X,(-1,X.shape[2]))
    pca=PCA(n_components=numComp,whiten=True)
    newX=pca.fit_transform(newX)
    newX=np.reshape(newX,(X.shape[0],X.shape[1],numComp))
    return newX

def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', True):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']



        label_values = ['Undefined', 'Trees', "Bare Soil", 'Bitumen', 'Self-Blocking Bricks']
            # , "Asphalt"
            # , "Meadows"]
        # label_values = ["Undefined", "Water", "Trees", "Asphalt",
        #                 "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
        #                 "Meadows", "Bare Soil"]
        ignored_labels = [0]
        rest_band = 102

    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']
        # H,W,B=img1.shape
        # img=np.zeros((H,W,B+5))
        # img[:,:,0:B]=np.copy(img1)
        # img[:, :,B] = np.copy(img1[:,:,B-1])
        # img[:,:,B+1:B+3]=np.copy(img[:,:,B-1:B])
        # img[:, :, B + 3:B + 5] = np.copy(img[:, :, B - 1:B])

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']
        # w,h=gt1.shape
        #
        # gt_ts=np.zeros_like(gt1)
        # gt_ts[(gt1 == 4)[:w, :h]] = 1
        # gt_ts[(gt1 == 6)[:w, :h]] = 2
        # gt_ts[(gt1 == 7)[:w, :h]] = 3
        # gt_ts[(gt1 == 8)[:w, :h]] = 4
        # gt_ts[(gt1 == 1)[:w, :h]] = 5
        # gt_ts[(gt1 == 2)[:w, :h]] = 6

        #label_values = ['Undefined', 'Trees', "Bare Soil", 'Bitumen', 'Self-Blocking Bricks']
            # , "Asphalt"
            # , "Meadows"]

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        # gt = np.copy(gt_ts)
        rest_band=108
        ignored_labels = [0]

    elif dataset_name == 'houston2013':
        # Load the image
        img = open_file(folder + 'Houston.mat')['Houston']
        # img_sampling=np.zeros_like(imgsd)
        # H,W,B=img1.shape
        # img=np.zeros((H,W,B+3))
        # img[:,:,0:B]=np.copy(img1)
        # img[:, :,B] = np.copy(img1[:,:,B-1])
        # img[:,:,B+1:B+3]=np.copy(img[:,:,B-1:B])

        # --------sampling to 48 bands--------
        # bands = 0
        # for i in range(48):
        #     img_sampling[:, :, i] = np.copy(imgsd[:, :, bands])
        #     bands += 3
        #
        # print(bands)
        # img = np.copy(img_sampling)



        rgb_bands = (59, 40, 13)

        # gt =open_file(folder + 'houston_tf13.npy')
        #gt = open_file(folder + 'houstontrain.tif')#['houston_gt']
        gt = open_file(folder + 'houston13_gt_new.mat')['houston_gt']

        label_values = ['Undefined', 'Healthy grass', 'Stressed grass', 'Synthetic grass',
                        'Trees', 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway',
                        'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']
        H,W=gt.shape

        # label_values = ['Undefined','Healthy grass','Stressed grass',
        #                 'Trees','Water','Residential','Commercial','Road']

        ignored_labels = [0]
        rest_band = 108

    elif dataset_name == 'houston2018':
        # Load the image
        img = open_file(folder + '2018_IEEE_GRSS_DF_Contest_Samples_TR.tif')[:,:,:-2]
        # H,W,B=img1.shape
        # img=np.zeros((H,W,B+1))
        # img[:,:,0:B]=np.copy(img1)


        rgb_bands = (19, 40, 13)

        gt = open_file(folder + 'houston2018_gt.mat')['houston2018_gt']

        # label_values = ['Undefined','Healthy grass','Stressed grass','trees','Water',
        #                 'Residential buildings','Non-residential buildings','Roads']
        label_values = ['Undefined', 'Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees',
                        'Deciduous trees','Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings', 'Roads',
                        'Sidewalks','Crosswalks', 'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots',
                        'Unpaved parking lots','Cars', 'Trains', 'Stadium seats']

        ignored_labels = [0]
        rest_band = 48

    elif dataset_name == 'YRE':
        # Load the image
        img = open_file(folder + 'data_hsi.mat')['data']
        # img=np.copy(img1.transpose((1,2,0)))

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'mask_test.mat')['mask_test']

        label_values = ['Undefined','Salt marsh','Acquaculture','Mud flat','Rice', 'Aquatic vegetation','Seep sea',
                        'Freshwater herbaceous marsh','Shallow sea','Reed', 'Pond' ,'Building','Suaeda salsa',
                        'Flood plain', 'River','Soybean', 'Broomcorn', 'Maize', 'Locust','Spartina', 'Tamarix']

        ignored_labels = [0]
        rest_band = 192

    elif dataset_name == 'Huanghe_obt':
        # Load the image
        img = open_file(folder + 'img_huanghe.tif')
        # img=np.copy(img1.transpose((1,2,0)))

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'gt_huanghe.tif')
        h,w = gt.shape
        gt[(gt==255)[:h,:w]]=17

        label_values = ['Undefined','Salt marsh','Acquaculture','Mud flat','Rice', 'Aquatic vegetation','Seep sea',
                        'Freshwater herbaceous marsh','Shallow sea','Reed', 'Pond' ,'Building','Suaeda salsa',
                        'Flood plain', 'River','Soybean', 'Broomcorn', 'Maize']

        ignored_labels = [0]
        rest_band = 192
    elif dataset_name == 'XiongAn':
        # Load the image
        img1 = h5py.File(folder + 'XiongAn.mat')['XiongAn'][:]
        img=np.copy(img1.transpose((1,2,0)))

        rgb_bands = (55, 41, 12)

        gt = h5py.File(folder + 'XiongAn_label.mat')['XiongAn_label'][:]

        label_values = ['Undefined','Salt marsh','Acquaculture','Mud flat','Rice', 'Aquatic vegetation','Seep sea',
                        'Freshwater herbaceous marsh','Shallow sea','Reed', 'Pond' ,'Building','Suaeda salsa',
                        'Flood plain', 'River','Soybean', 'Broomcorn', 'Maize', 'Locust','Spartina', 'Tamarix']

        ignored_labels = [0]
        rest_band = 192
    elif dataset_name == 'YC':
        # Load the image
        img = open_file(folder + 'YC_hsi.mat')['YC_hsi']
        # img=np.copy(img1.transpose((1,2,0)))

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'mask_test.mat')['mask_test']
        print(np.max(gt))

        label_values = ['Undefined','Sea','Offshore area','Salt field','Pond', ' Partina anglica','Mudflats',
                        'Aquaculture pond','Paddy filed','Estuarine area', 'River' ,'Woodland','Barren',
                        'Building', 'Fallow land','Rainfed cropland', 'Suaeda salsa', 'Irrigation canal', 'Phragmites']

        ignored_labels = [0]
        rest_band = 192

    elif dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-Cleantill", "Corn-Cleantill-EW",
                        "Orchard","Soybean-CleanTill", "Soybeans-CleanTill-EW",
                        "Wheat",'Salt marsh','Acquaculture','Mud flat','Rice', 'Aquatic vegetation','Seep sea',
                        'Freshwater herbaceous marsh','River', 'Broomcorn']

        ignored_labels = [0]
        rest_band=192
    elif dataset_name == 'HangZhou':
        # Load the image
        img = open_file(folder + '2.mat')
        img = img['DataCube2']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + '2_gt.mat')['gt2']
        label_values = ["Undefined", "Water", "Land/Building", "Plant"]

        ignored_labels = [0]
        rest_band = 192

    elif dataset_name == 'ShangHai':
        # Load the image
        img = open_file(folder + '1.mat')
        img = img['DataCube1']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + '1_gt.mat')['gt1']
        label_values = ["Undefined", "Water", "Land/Building", "Plant"]

        ignored_labels = [0]
        rest_band = 192

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
    #data = preprocessing.scale(data)
    data  = preprocessing.minmax_scale(data)
    img = data.reshape(img.shape)
    return img, gt, label_values, ignored_labels, rgb_bands, palette,rest_band


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data,data_name, rest_band,gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = data_name
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.reratio=hyperparams['re_ratio']
        self.counts=0        # self.flip_augmentation = hyperparams['flip_augmentation']
        # self.radiation_augmentation = hyperparams['radiation_augmentation']
        # self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        # self.model_name=hyperparams['model']
        self.spectral_fusion=hyperparams['spectral_fusion']
        self.rest_band=rest_band
        supervision = 'full'
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        # center_Data = self.data[x,y]
        label = self.label[x1:x2, y1:y2]

        # if self.reratio>0:
        #     H,W,C=self.data.shape
        #     orignal_H=int(H/(self.reratio+1))
        #     if x< 2*orignal_H and x >= orignal_H:
        #         # Perform data augmentation (only on 2D patches)
        #         data, label = self.flip(data, label)
        #     if x>=2*orignal_H and x<3*orignal_H:
        #         data = self.mixture_noise(data, label)
        #     if x>=3*orignal_H:
        #         data = self.radiation_noise(data)



        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # center_Data = np.asarray(np.copy(center_Data), dtype='float32')
        # data=(np.copy(data1[0:144,:,:])).reshape((4,96,96))
        label = np.asarray(np.copy(label), dtype='int64')


        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # center_Data = torch.from_numpy(center_Data)

        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN

        if self.spectral_fusion:
            data1 = np.copy(data[0:self.rest_band, :, :])
            C1, H1, W1 = data1.shape
            win_size = int(((C1 / 3) ** 0.5) * H1)
            lter_num = int(win_size / H1)
            spec_num = 0
            data_spectral = np.zeros((3, win_size, win_size), dtype='float32')
            for i in range(lter_num):
                for j in range(lter_num):
                    data_spectral[:, (H1 * i):(H1 * (i + 1)), (W1 * j):(W1 * (j + 1))] = np.copy(data1[
                                                                                                 (3 * (spec_num)):(3 * (
                                                                                                             spec_num + 1)),
                                                                                                 :, :])
                    # print((H1*i),(H1*(i+1)),(W1*j),(W1*(j+1)))
                    spec_num += 1
            data = np.copy(data_spectral)
            data= transforms.Resize([5,256,256])



            # viz = visdom.Visdom(env=self.DATASET + ' ' + self.MODEL)
            # display_predictions(np.transpose(data[0:3,:,:],(1,2,0)),viz)
            # display_predictions(np.transpose(data1[0:3, :, :], (1, 2, 0)), viz)
            # quit()


        return data , label


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
        train_indices = tuple([list(t) for t in zip(*train_indices)])
        test_indices = tuple([list(t) for t in zip(*test_indices)])
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
        print("Sampling {} with train size = {}".format(mode, train_size))
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features

            train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    elif ext == '.npy':
        img = np.load(dataset)
        H,W=img.shape
        return img
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def principal_component_extraction(spectral_original, variance_required):
    ## Variable List
    ## spectral_original: The original non reduced image
    ## variance_required: The required variance  ratio from 0 to 1

    ## Output list
    ## spectral_pc_final: The dimensional reduces image

    # 2d reshape
    spectral_2d = spectral_original.reshape(
        (spectral_original.shape[0] * spectral_original.shape[1], spectral_original.shape[2]))
    # Feature scaling preprocessing step
    spectral_2d = preprocessing.scale(spectral_2d)

    if (spectral_2d.shape[1] < 100):
        pca = PCA(n_components=spectral_2d.shape[1])
    else:
        pca = PCA(n_components=100)
    spectral_pc = pca.fit_transform(spectral_2d)
    explained_variance = pca.explained_variance_ratio_

    if (np.sum(explained_variance) < variance_required):
        raise ValueError("The required variance was too high. Values should be between 0 and 1.")

    # Select the number of principal components that gives the variance required
    explained_variance_sum = np.zeros(explained_variance.shape)
    sum_ev = 0
    component_number = 0
    for i in range(explained_variance.shape[0]):
        sum_ev += explained_variance[i]
        if (sum_ev > variance_required and component_number == 0):
            component_number = i + 1
        explained_variance_sum[i] = sum_ev

    # Removed the unnecessary components and reshape in original 3d form
    spectral_pc = spectral_pc[:, :component_number]
    spectral_pc_final = spectral_pc.reshape((spectral_original.shape[0], spectral_original.shape[1], component_number))

    return spectral_pc_final

def HSI_dataloder(cfg):
    img_ori, gtSD, LABEL_VALUES, IGNORED_LABELS, RGB_BANDSSD, palette, rest_bandSD = get_dataset(cfg.source_HSI,
                                                                                                          cfg.folder)


    # img_ori = apply_pca(img_ori,numComp=30)
    H_SD, W_SD, Channel_SD = img_ori.shape
    img1 = img_ori.reshape(-1, Channel_SD)
    stand_img1 = sklearn.preprocessing.StandardScaler().fit_transform(img1)
    imageSD = stand_img1.reshape(img_ori.shape[:3])
    # print(Channel_SD)
    # imageSD = np.copy(img_ori)
    # imageSD = np.copy(stand_img1)
    # imageTD, gtTD, _, _, RGB_BANDSTD, _, rest_bandTD = get_dataset(cfg.target_HSI,
    #                                                                         cfg.folder)
    # imgSD_PCA = principal_component_extraction(imageSD, 0.998)
    # imgTD_PCA = principal_component_extraction(imageTD, 0.998)


    # H_SD, H_TD, Channel_TD = imageTD.shape

    Inchannel_pca = Channel_SD
    imgSD = np.copy(imageSD[:, :, 0:Inchannel_pca])
    n_classes = int(np.max(gtSD) + 1)

    # imgTD = np.copy(imageTD[:, :, 0:Inchannel_pca])

    # pseudo_tgt = np.load('pseudo_labe_houston13_18.npy')

    # src_img, src_labels =get_dataset(args.source, path=args.db_path)
    # tgt_img, tgt_labels = get_dataset(args.target, path=args.db_path)

    # sparse_ground_truth = sparseness_operator(gtSD, cfg.train_num)
    if cfg.pretrain:
        train_gt = np.ones_like(gtSD)
    elif cfg.disjoint:

        if cfg.source_HSI == 'houston2013':
            train_gt = open_file('./Datasets/houston2013/disjoint_train.mat')['training_gt']
        elif cfg.source_HSI == 'PaviaU':
            train_gt = open_file('./Datasets/PaviaU/disjoint_training.mat')['trainingMap']
        elif cfg.source_HSI == 'IndianPines':
            train_gt = open_file('./Datasets/IndianPines/disjoint_train.mat')['trainingMap']
        elif cfg.source_HSI == 'YRE':
            train_gt = open_file('./Datasets/YRE/mask_train.mat')['mask_train']
        elif cfg.source_HSI == 'YC':
            train_gt = open_file('./Datasets/YC/mask_train.mat')['mask_train']
        else:
            raise ValueError('{} dataset has not the disjoint version'.format(cfg.source_HSI))
            # train_gt, _ = sample_gt(train_all, cfg.training_sample, mode='random')
        test_gt = np.copy(gtSD)
        test_gt[(train_gt != 0)[:H_SD, :W_SD]] = 0
        # io.savemat('PaviaU_test_gt.mat',{'test_gt':test_gt})
        # quit()
    else:
        train_gt, test_gt = sample_gt(gtSD, cfg.training_sample, mode='random')



    _,val_gt=sample_gt(train_gt, 0.9, mode='random')

        # train_gt=np.copy(sparse_ground_truth)
        # superpixel_map = np.load('./superpixel/houston2018_superpixel_9000.npy')
        # train_pse = Accuracy_parall(superpixel_map, sparse_ground_truth)
        # train_gt, _ = sample_gt(train_pse, cfg.training_sample, mode='random')
        # caculate_gt = np.copy(gtSD)
        # caculate_gt[(train_gt == 0)[:H_SD, :W_SD]] = 0
        # train_gt[(caculate_gt == 0)[:H_SD, :W_SD]] = 0
        # acc_sup = 1 - (np.count_nonzero(train_gt - caculate_gt) / np.count_nonzero(train_gt))
        # print('superpixel acc :', acc_sup)






    # train_gt, val_gt = sample_gt(train_gt, 0.9, mode='random')

    # repeat_time = np.count_nonzero(test_gt) // np.count_nonzero(train_gt)
    # img_repeat = np.copy(imgSD)
    # gtSD_repeat = np.copy(train_gt)
    # if repeat_time==-0.1:
    #     for i in range(repeat_time - 1):
    #         img_repeat = np.concatenate((img_repeat, imgSD))
    #         gtSD_repeat = np.concatenate((gtSD_repeat, train_gt))


    for i in range(n_classes-1):
        count_class = np.copy(train_gt)
        test_count = np.copy(test_gt)
        # sparse_class=np.copy(sparse_ground_truth)

        count_class[(train_gt != i + 1)] = 0
        # sparse_class[(sparse_ground_truth != i + 1)[:H_SD, :W_SD]] = 0
        class_num=np.count_nonzero(count_class)

        test_count[test_gt!= i+1] = 0




        print(LABEL_VALUES[i + 1],':', class_num,np.count_nonzero(test_count))


    print(np.count_nonzero(train_gt),np.count_nonzero(test_gt))

    viz = visdom.Visdom(env=cfg.source_HSI + '' + cfg.model)
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", n_classes - 1)):
        if k == 9:
            palette[k + 1] = tuple(np.asarray([255, 255, 255], dtype='uint8'))
        else:
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    invert_palette = {v: k for k, v in palette.items()}
    # quit()






    hyperparams = vars(cfg)
    hyperparams['ignored_labels'] = IGNORED_LABELS
    hyperparams['center_pixel'] = True
    hyperparams['HSI_class']=n_classes
    hyperparams['inchanl']=Inchannel_pca
    hyperparams['label_values']=LABEL_VALUES



    if hyperparams['pretrain']:
        src_trainset = HyperX(imgSD, hyperparams['source_HSI'], rest_bandSD, train_gt, **hyperparams)
    else:
        src_trainset = HyperX(imgSD, hyperparams['source_HSI'], rest_bandSD, train_gt, **hyperparams)


    src_valset = HyperX(imgSD, hyperparams['source_HSI'], rest_bandSD, test_gt, **hyperparams)
    valset = HyperX(imgSD, hyperparams['source_HSI'], rest_bandSD, val_gt, **hyperparams)

    display_predictions(convert_to_color_(train_gt,palette=palette), viz,
                        caption="Train_gt")
    display_predictions(convert_to_color_(test_gt, palette=palette), viz,
                        caption="Test_gt")
    # quit()

    return src_trainset,src_valset,gtSD,test_gt,valset, imgSD


class LMMDLoss(nn.Module):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 gamma=1.0, max_iter=1000, **kwargs):
        '''
        Local MMD
        '''
        super(LMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.gamma=gamma
        self.max_iter=max_iter
        self.curr_iter = 0


        self.num_class = num_class

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def forward(self, source, target, source_label, target_logits):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")

        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            weight_ss = torch.from_numpy(weight_ss).cuda()  # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.guassian_kernel(source, target,
                                           kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).cuda()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            loss = loss * lamb
            return loss

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label]  # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum  # label ratio

        # Pseudo label
        target_label = target_logits.cpu().data.max(1)[1].numpy()

        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class):  # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1)  # (B, 1)

                ss = np.dot(s_tvec, s_tvec.T)  # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


def Accuracy_parall(superpixel_label,gt):
    ground_truth = np.copy(gt)
    H, W = gt.shape
    superpixel_map=np.copy(superpixel_label)
    # superpixel_map[(gt == 0)[:H, :W]] = 0


    superpixel_gt=np.zeros((H,W))
    count=0

    for i in range(H):
        for j in range(W):
            if ground_truth[i,j]!=0:
                sup_label=superpixel_map[i,j]
                # ideal_label=best_sample(superpixel_map,gt,img,sup_label)
                superpixel_gt[(superpixel_map==sup_label)]=ground_truth[i,j]

                ground_truth[(superpixel_map==sup_label)[:H,:W]]=0
                count+=1

    # all_num1 = np.count_nonzero(gt)
    # superpixel_gt[(gt == 0)[:H, :W]] = 0
    print('number is all',count)
    #super_num1 = np.count_nonzero(superpixel_gt)



    return superpixel_gt

def sparseness_operator(ground_truth,number_of_samples):
    sparse_ground_truth =  np.reshape(np.copy(ground_truth) , -1)
    # HOW MANY OF EACH LABEL DO WE WANT
    number_of_classes = np.amax(ground_truth)
    for i in range(number_of_classes):
        index = np.where(sparse_ground_truth == i+1)[0]
        bing=index.shape[0]
        if(index.shape[0] < number_of_samples):
            index = np.random.choice(index,index.shape[0],replace = False)
        else:
            index = np.random.choice(index,index.shape[0] - number_of_samples,replace = False)
        index = np.sort(index)
        sparse_ground_truth[index] = 0
    sparse_ground_truth = sparse_ground_truth.reshape((ground_truth.shape))
    # sio.savemat('Houston_train_3.mat',mdict={'houston':sparse_ground_truth})

    return sparse_ground_truth

def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    # classifier.eval()
    patch_size =hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['nb_classes']

    kwargs = {'step': 1, 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))
    img = np.pad(img,((patch_size // 2,patch_size // 2),(patch_size // 2,patch_size // 2),(0,0)),"symmetric")

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = (np.copy(data)).transpose(0, 3, 1, 2)
                # data_center = data[:,:,patch_size//2,patch_size//2]


                # number_batch,_,_,_=data1.shape
                # data = np.copy((data1[:, 0:144, :, :]).reshape((number_batch, 4, 96, 96)))
                data = torch.from_numpy(data)
                # data_center = torch.from_numpy(data_center)

                if hyperparams['spectral_fusion']:
                    data1 = np.copy(data[:,0:hyperparams['rest_band'], :, :])
                    B1,C1, H1, W1 = data1.shape
                    win_size = int(((C1 / 3) ** 0.5) * H1)
                    lter_num = int(win_size / H1)
                    spec_num = 0
                    data_spectral = np.zeros((B1,3, win_size, win_size), dtype='float32')
                    for i in range(lter_num):
                        for j in range(lter_num):
                            data_spectral[:,:, (H1 * i):(H1 * (i + 1)), (W1 * j):(W1 * (j + 1))] = np.copy(data1[:,
                                                                                                         (3 * (spec_num)):(3 * (
                                                                                                                 spec_num + 1)),:, :])
                            # print((H1*i),(H1*(i+1)),(W1*j),(W1*(j+1)))
                            spec_num += 1
                    data = np.copy(data_spectral)
                    data = torch.from_numpy(data)
                elif hyperparams['model']=='hamida':
                    data = data.unsqueeze(1)
                # data_transforms=transforms.Resize([224,224])
                # data=data_transforms(data)


            indices = [b[1:] for b in batch]
            data = data.to(device)
            # data_center = data_center.to(device)

            # output = net(data)
            outputs = net(data)
            # output = classifier.predict(outputs[0])
            output = outputs[1]
            # output1 = classifier.predict(outputs[0])
            # output = (output1 + outputs[1]) / 2
            # output = (output1 + output2) / 2
            # output = classifier.predict(outputs[0])


            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x, y] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs

def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    #target = target[ignored_mask] -1
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)

    accuracy = sum([cm[x][x] for x in range(len(cm))])

    class_accuracy = [cm[x][x] / sum(cm[x]) for x in range(len(cm))]
    results['class_acc'] = class_accuracy
    accuracy *= 100 / float(total)
    results['average_acc'] = np.mean(class_accuracy[1:])
    # print(results['average_acc'])
    # quit()

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results



    # Generate color palette


# def convert_to_color(x):
#     return convert_to_color_(x, palette=palette)

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})