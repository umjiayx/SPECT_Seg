import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import torch
import datasets
import torch.nn.functional as F
# from julia import Julia
# jl = Julia(compiled_modules=False)
# from julia.AVSfldIO import fld_read
from grid_sample import *
import matplotlib.patches as patches
from utils import init_env, find_contour, get_bounding_box_center


# import matlab.engine
# eng = matlab.engine.start_matlab()

class mydataloader:
    def __init__(self, datasetdir = '.', mode = 'train', 
                 known_xtrue = True, verbose = True) -> None:
        # torch.set_float32_matmul_precision('highest')
        self.imgsize = 256
        self.verbose = verbose
        assert mode in {'train', 'test'}
        if mode == 'train':
            self.dir = os.path.join(datasetdir, 'train')  
        else:
            self.dir = os.path.join(datasetdir, 'test')
        self.known_xtrue = known_xtrue
                  
        if self.known_xtrue:
            self.meta = {'spect': [], 'xtrue': [], 'seg':[]}
        else:
            self.meta = {'spect': []}
        self.readfldfiles()
        # normalize xtrue
    def convert_data(self):
        for i in range(len(self.meta['spect'])):
            maxofspect = torch.max(self.meta['spect'][i])
            if self.verbose:
                print('max of spect: ', maxofspect)
            if self.known_xtrue:
                maxofxtrue = torch.max(self.meta['xtrue'][i])
                if self.verbose:
                    print('max of xtrue, before normalization: ', maxofxtrue)
                self.meta['xtrue'][i] = self.meta['xtrue'][i] / maxofxtrue * 255.0
                if self.verbose:
                    print('max of xtrue, after normalization: ', torch.sum(self.meta['xtrue'][i]))
            max_of_spect = torch.max(self.meta['spect'][i])
            self.meta['spect'][i] = self.meta['spect'][i] / max_of_spect * 255.0
            if self.verbose:
                print('max of spect after normalization: ', torch.max(self.meta['spect'][i]))     
                                 
        self.meta['spect'] = torch.stack(self.meta['spect']).permute(0,3,1,2).reshape(-1,self.imgsize,self.imgsize)

        spect_temp = torch.clone(self.meta['spect'])
        if self.known_xtrue:
            self.meta['xtrue'] = torch.stack(self.meta['xtrue']).permute(0,3,1,2).reshape(-1,self.imgsize,self.imgsize)
            self.meta['seg'] = torch.stack(self.meta['seg']).permute(0,3,1,2).reshape(-1,self.imgsize,self.imgsize)
            xtrue_temp = torch.clone(self.meta['xtrue'])
            seg_temp = torch.clone(self.meta['seg'])
            
            for imgidx in range(self.meta['seg'].shape[0]):
                seg = self.meta['seg'][imgidx, :, :]
                filled_segs = find_contour(seg)
                if len(filled_segs) > 0:
                    print(f'{imgidx} seg is separate into: ', len(filled_segs))
                    seg_temp[imgidx, :, :] = filled_segs[0].to(self.meta['seg'].dtype)
                    for segidx in range(1, len(filled_segs)):
                        seg_temp = torch.cat((seg_temp, 
                                            filled_segs[segidx].unsqueeze(0).to(self.meta['seg'].dtype)),
                                            dim=0)
                        spect_temp = torch.cat((spect_temp, 
                                                self.meta['spect'][imgidx, :, :].unsqueeze(0)),
                                                dim=0)
                        xtrue_temp = torch.cat((xtrue_temp, 
                                                self.meta['xtrue'][imgidx, :, :].unsqueeze(0)),
                                                dim=0)
            self.meta['xtrue'] = torch.clone(xtrue_temp)
            self.meta['seg'] = torch.clone(seg_temp)
        
        self.meta['spect'] = torch.clone(spect_temp)
                    
        if self.verbose:
            print('spect shape: ', self.meta['spect'].shape)
            print('spect maximum: ', torch.max(self.meta['spect']))
            print('spect average: ', torch.mean(self.meta['spect']))
        if self.known_xtrue:
            if self.verbose:
                print('xtrue shape: ', self.meta['xtrue'].shape)
                print('seg shape: ', self.meta['seg'].shape)
                print('xtrue maximum: ', torch.max(self.meta['xtrue']))
                print('xtrue mean: ', torch.mean(self.meta['xtrue']))
                print('seg maximum: ', torch.max(self.meta['seg']))
        
                
    def readfldfiles(self):
        path_list= []
        file_list = []
        for root, _, files in os.walk(self.dir):
            path_list.append(root)
            files = [fi for fi in files if fi.endswith(".fld")]
            file_list.append(files)
        file_num = len(path_list) - 1
        spect_str = 'spect128wSC'
        xtrue_str = 'xtrue'
        seg_str = 'lesionbigmask'
        
        for i in range(file_num):
            fldfiles = file_list[i + 1]
            if self.verbose:
                print('load data from {}'.format(path_list[i + 1]))
            for s in fldfiles:
                if spect_str in s:
                    if self.verbose:
                        print('load spect map from {}'.format(os.path.join(path_list[i + 1], s)))
                    spect = torch.tensor(fld_read(os.path.join(path_list[i + 1], s)), dtype=torch.float)
                    spect = resize128to256(spect)[:,:,20:60]
                    # spect = m(spect.unsqueeze(0).unsqueeze(0)).squeeze()
                    self.meta['spect'].append(spect)
                if self.known_xtrue:
                    if xtrue_str in s:
                        if self.verbose:
                            print('load true map from {}'.format(os.path.join(path_list[i + 1], s)))
                        xtrue = torch.tensor(fld_read(os.path.join(path_list[i + 1], s)), dtype=torch.float)
                        xtrue = resize512to256(xtrue)[:,:,20:60]
                        # xtrue = m(xtrue.unsqueeze(0).unsqueeze(0)).squeeze()
                        self.meta['xtrue'].append(xtrue)
                    if seg_str in s:
                        if self.verbose:
                            print('load seg map from {}'.format(os.path.join(path_list[i + 1], s)))
                        seg = torch.tensor(fld_read(os.path.join(path_list[i + 1], s)), dtype=torch.float)
                        seg = resize512to256(seg).bool().float()[:,:,20:60]
                        # seg = m(seg.unsqueeze(0).unsqueeze(0)).squeeze()
                        self.meta['seg'].append(seg)
            if self.verbose:
                print('data load {} / {} finished!'.format(i + 1, file_num))
                
def create_dataset(images, labels):
    dataset = datasets.Dataset.from_dict({"image": images,
                                "label": labels})
    dataset = dataset.cast_column("image", datasets.Image())
    dataset = dataset.cast_column("label", datasets.Image())

    return dataset

def read_mat_files(path):
    spect = sio.loadmat(os.path.join(path, 'spect_128.mat'))['spect_128']
    seg = sio.loadmat(os.path.join(path, 'lesionbigmask_128.mat'))['seg_128']
    return spect, seg

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 144/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":
    loader = mydataloader(datasetdir='/home/zhonglil/ondemand/data/sys/myjobs/default/SAM_seg/', mode='test')
    loader.convert_data()
    init_env(seed_value=42)
    bbox_threshold = 20
    # path = './y90-data/test/y90res02'
    # spect, seg = read_mat_files(path=path)
    # spect = np.transpose(spect, [2,0,1]) / np.amax(spect) * 255.0
    # print('maximum of spect: ', np.amax(spect))
    # seg = np.transpose(seg, [2,0,1])
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # num_of_img x dim_of_img
    dataset = create_dataset(images=loader.meta['spect'], labels=loader.meta['seg'])
    # processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
    # model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
    # plt.figure()
    fig, axes = plt.subplots()
    idx = 17
    axes.imshow(np.array(dataset[idx]["image"]))
    ground_truth_seg = np.array(dataset[idx]["label"])
    bbox, _ = get_bounding_box_center(ground_truth_seg, bbox_threshold)
    _, center = get_bounding_box_center(ground_truth_seg, bbox_threshold=1)
    print('bbox values: ', bbox)
    print('center values: ', center)
    show_mask(ground_truth_seg, axes)
    # Create a Rectangle patch
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                             linewidth=3, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    axes.add_patch(rect)
    axes.title.set_text(f"Ground truth mask")
    axes.axis("off")
    plt.savefig('testimg.png')
    plt.show()
    # plt.figure()
    # plt.imshow(loader.meta['seg'][idx])
    # plt.savefig('testimg_seg.png')
    # plt.show()