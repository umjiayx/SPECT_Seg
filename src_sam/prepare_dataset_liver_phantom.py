import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import torch
import datasets
from julia import Julia
jl = Julia(compiled_modules=False)
from julia.AVSfldIO import fld_read
from grid_sample import *
import matplotlib.patches as patches
from utils import init_env, find_contour, get_bounding_box_center


class liverphantomdataloader:
    def __init__(self, datasetdir = '.', verbose = True) -> None:
        # torch.set_float32_matmul_precision('highest')
        self.imgsize = 256
        self.verbose = verbose
        self.dir = datasetdir
                  
        self.meta = {'spect': [], 'seg':[]}
        spect = torch.tensor(fld_read(os.path.join(self.dir, 'xcnn-sc-iter15-idx47-resize256.fld')), 
                             dtype=torch.float)
        self.meta['spect'].append(spect)
        
        seg = torch.tensor(fld_read(os.path.join(self.dir, 'liver-phantom-mask-idx47-resize256.fld')), 
                           dtype=torch.float)
        seg = seg.bool().float()
        self.meta['seg'].append(seg)
        
        maxofspect = torch.max(self.meta['spect'][0])
        if self.verbose:
            print('max of spect: ', maxofspect)
        self.meta['spect'][0] = self.meta['spect'][0] / maxofspect * 255.0
        self.meta['spect'] = self.meta['spect'][0].unsqueeze(0)
        spect_temp = torch.clone(self.meta['spect'])
        self.meta['seg'] = self.meta['seg'][0].unsqueeze(0)
        seg_temp = torch.clone(self.meta['seg'])
        
        seg = self.meta['seg'][0, :, :]
        print(seg.shape)
        filled_segs = find_contour(seg)
        if len(filled_segs) > 0:
            print('seg is separate into: ', len(filled_segs))
            seg_temp[0, :, :] = filled_segs[0].to(self.meta['seg'].dtype)
            for segidx in range(1, len(filled_segs)):
                seg_temp = torch.cat((seg_temp, 
                                    filled_segs[segidx].unsqueeze(0).to(self.meta['seg'].dtype)),
                                    dim=0)
                spect_temp = torch.cat((spect_temp, 
                                        self.meta['spect'][0, :, :].unsqueeze(0)),
                                        dim=0)
        self.meta['spect'] = torch.clone(spect_temp)
        self.meta['seg'] = torch.clone(seg_temp)
        if self.verbose:
            print('spect shape: ', self.meta['spect'].shape)
            print('spect maximum: ', torch.max(self.meta['spect']))
            print('spect average: ', torch.mean(self.meta['spect']))
            print('seg shape: ', self.meta['seg'].shape)
            print('seg maximum: ', torch.max(self.meta['seg']))
                
                
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
    loader = liverphantomdataloader(datasetdir='../y90-data-wden/liver-phantom/')
    init_env(seed_value=42)
    bbox_threshold = 10
    dataset = create_dataset(images=loader.meta['spect'], labels=loader.meta['seg'])
    fig, axes = plt.subplots()
    idx = 2
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
    plt.savefig('testimg-phantom.png')
    plt.show()
    # plt.figure()
    # plt.imshow(loader.meta['seg'][idx])
    # plt.savefig('testimg_seg.png')
    # plt.show()