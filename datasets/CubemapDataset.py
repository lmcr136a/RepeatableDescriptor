"""

"""
import os
import torch
import cv2
import numpy as np
import torch.utils.data as data
import scipy.io as io
from pathlib import Path
from imageio import imread
from utils.tools import dict_update
from scipy.spatial.transform import Rotation as R


def load_as_float(path):
    return imread(path).astype(np.float32)/255

class CubemapDataset(data.Dataset):
    default_config = {
        'dataset': 'cubemap',  # or 'hpatches' or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': False
        }
    }

    def __init__(self, transform=None, **config):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.files = self._init_dataset(**self.config)
        sequence_set = []
        for (img, img_warped) in zip(self.files['image_paths'], self.files['warped_image_paths']):
            sample = {'image': img, 'warped_image': img_warped}
            sequence_set.append(sample)
        self.samples = sequence_set
        self.transform = transform
        if config['preprocessing']['resize']:
            self.sizer = np.array(config['preprocessing']['resize'])
        else:
            self.sizer = 'No resize'

        self.custom_dir = 'zzours'
        self.RTfile = io.loadmat(os.path.join(self.custom_dir, 'RTDB.mat'))
        self.Kptfile = io.loadmat(os.path.join(self.custom_dir, 'KeyPts2D3D.mat'))
        pass

    def __getitem__(self, index):
        """

        :param index:
        :return:
            image:
                tensor (1,H,W)
            warped_image:
                tensor (1,H,W)
        """
        def _read_image(path):
            input_image = cv2.imread(path)
            return input_image

        def _preprocess(image):
            if type(self.sizer) == str and self.sizer == 'No resize':
                h, w, c = image.shape
                if c == 3:
                    self.sizer = np.array([h, w])
                elif h == 3:
                    self.sizer = np.array([w, c])
            s = max(self.sizer /image.shape[:2])
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[:int(self.sizer[0]/s),:int(self.sizer[1]/s)]
            image = cv2.resize(image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            if image.ndim == 2:
                image = image[:,:, np.newaxis]
            if self.transform is not None:
                image = self.transform(image)
            return image

        sample = self.samples[index]
        # zzours/cubemaps/setup019/th5.png

        name = sample['image'][len(self.custom_dir)+1:]
        name_w = sample['warped_image'][len(self.custom_dir)+1:]
        setupnums = name.split('setup')[-1].split('/')[0][1:]
        ply_path = os.path.join(self.custom_dir, 'ply', f'FF230120_Setup{setupnums}.ply')
        cmpath = list(map(lambda x: x.replace(" ", ""), self.RTfile['cubemap_path_unit5']))
        idx = cmpath.index(name)
        idx_w = cmpath.index(name_w)
        if idx is None or idx_w is None:
            print()
            print(name)
            print(name_w)
            print(idx, idx_w)
            raise
        T, T_w = self.RTfile['T_unit5'][idx], self.RTfile['T_unit5'][idx_w]
        R, R_w = self.RTfile['R_unit5'][idx], self.RTfile['R_unit5'][idx_w]
        kpts2D, kpts3D = self.Kptfile[self.custom_dir+'/'+name+'2Dkpts'], self.Kptfile[self.custom_dir+'/'+name+'_3Dkpts']
        kpts2D_w, kpts3D_w = self.Kptfile[self.custom_dir+'/'+name_w+'2Dkpts'], self.Kptfile[self.custom_dir+'/'+name_w+'_3Dkpts']
        if 0:
            # setupnums = list(map(lambda x: x.split('setup')[-1].split('/')[0][1:], sample['image']))
            # ply_path = list(map(lambda x: os.path.join(self.custom_dir, 'ply', 'FF230120_Setup{x}.ply'), setupnums))
            # print(sample['image'])
            # T = list(map(lambda x: self.RTfile['T_unit5'][np.where(self.RTfile['cubemap_path_unit5'] == x)[0]], sample['image']))
            # T_w = list(map(lambda x: self.RTfile['T_unit5'][np.where(self.RTfile['cubemap_path_unit5'] == x)[0]], sample['warped_image']))
            # R = list(map(lambda x: self.RTfile['R_unit5'][np.where(self.RTfile['cubemap_path_unit5'] == x)[0]], sample['image']))
            # R_w = list(map(lambda x: self.RTfile['R_unit5'][np.where(self.RTfile['cubemap_path_unit5'] == x)[0]], sample['warped_image']))
            pass 

        image_original = _read_image(sample['image'])
        image = _preprocess(image_original)
        warped_image = _preprocess(_read_image(sample['warped_image']))
        to_numpy = False
        if to_numpy:
            image, warped_image = np.array(image), np.array(warped_image)

        T, T_w, R, R_w = np.array(T), np.array(T_w),np.array(R),np.array(R_w)
        max_kpts_num = 2650 # max kpts number
        k2Darray, k3Darray, k2Darray_w, k3Darray_w = np.zeros((max_kpts_num, 2)), np.zeros((max_kpts_num, 3)), np.zeros((max_kpts_num, 2)), np.zeros((max_kpts_num, 3))
        k2Darray[:len(kpts2D)] = kpts2D
        k3Darray[:len(kpts3D)] = kpts3D
        k2Darray_w[:len(kpts2D_w)] = kpts2D_w
        k3Darray_w[:len(kpts3D_w)] = kpts3D_w
        sample = {'image': image, 'warped_image': warped_image,
                  'ply_path': ply_path, 'R': R, 'T': T, 'R_w':R_w, 'T_w':T_w,
                  'img_path': sample['image'], 'img_path_w': sample['warped_image'],
                  'kpts2D': k2Darray, 'kpts3D': k3Darray, 'kpts2D_w': k2Darray_w, 'kpts3D_w': k3Darray_w, 
                  }
        return sample

    def __len__(self):
        return len(self.samples)

    def _init_dataset(self, **config):
        dataset_folder = 'zzours/cubemaps'
        base_path = Path(dataset_folder)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:  # setups
            # if config['alteration'] == 'i' and path.stem[0] != 'i':
            #     continue
            # if config['alteration'] == 'v' and path.stem[0] != 'v':
            #     continue
            num_images = 5
            file_ext = '.png'
            for angle in range(0, 360, 5):
                near_angles = get_near_angle_filenames(angle)
                for near_angle in near_angles:
                    image_paths.append(str(Path(path, f"th{angle}{file_ext}")))
                    warped_image_paths.append(str(Path(path, f"th{near_angle}{file_ext}")))
            # for i in range(2, 2 + num_images):
            #     image_paths.append(str(Path(path, "1" + file_ext)))
            #     warped_image_paths.append(str(Path(path, str(i) + file_ext)))
            #     homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        # if config['truncate']:
        #     image_paths = image_paths[:config['truncate']]
        #     warped_image_paths = warped_image_paths[:config['truncate']]
        #     homographies = homographies[:config['truncate']]
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 }
        return files


def get_near_angle_filenames(angle, unit=5, num=15):
    d = np.array(list(range(unit, unit*num+unit, unit)))
    d = np.concatenate([d, -d])
    d += angle
    def clip(ang):
        if ang > 359:
            ang -= 360
        elif ang < 0:
            ang = 360+ang
        return ang
    d = np.array(list(map(clip, d)))
    return d
