import os
import torch 
import numpy as np
from utils import init_env, find_contour
import SimpleITK as sitk
# from julia import Julia
# jl = Julia(compiled_modules=False)
# from julia.AVSfldIO import fld_read, fld_write
from prepare_dataset_zhonglin import zhonglindataloader
import config

def read_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def write_image(image, path):
    sitk.WriteImage(sitk.GetImageFromArray(image), path)
    
if __name__ == "__main__":
    loader = zhonglindataloader(datasetdir=config.path_to_test_data, delete_zeros=False)
    result_path = config.root_dir + '-zhonglin-bbox=10-prompt=box-1102'
    outputseg = read_image(os.path.join(result_path, 'outputseg.nii.gz'))
    trueseg = read_image(os.path.join(result_path, 'trueseg.nii.gz'))
    # outputseg = fld_read(os.path.join(result_path, 'outputseg.fld'))
    # trueseg = fld_read(os.path.join(result_path, 'trueseg.fld'))
    loader.meta['seg'] = loader.meta['seg'].permute(0,3,2,1).reshape(-1,loader.imgsize,loader.imgsize)
    
    put_back_idx = loader.meta['seg'].shape[0]
    print("test:",put_back_idx)
    for imgidx in range(loader.meta['seg'].shape[0]):
        seg = loader.meta['seg'][imgidx, :, :]
        filled_segs = find_contour(seg)
        if len(filled_segs) > 0:
            print(f'put {put_back_idx} back into {imgidx}')
            for j in range(1, len(filled_segs)):
                outputseg[imgidx] += outputseg[put_back_idx]
                trueseg[imgidx] += trueseg[put_back_idx]
                put_back_idx += 1
    outputseg = outputseg[:loader.meta['seg'].shape[0], :, :]
    trueseg = trueseg[:loader.meta['seg'].shape[0], :, :]
    print('outputseg shape: ', outputseg.shape)
    print('trueseg shape: ', trueseg.shape)
    # fld_write(os.path.join(result_path, 'outputseg_putback.fld'), outputseg)
    # fld_write(os.path.join(result_path, 'trueseg_putback.fld'), trueseg)
    write_image(outputseg, os.path.join(result_path, 'outputseg_putback.nii.gz'), )
    write_image(trueseg, os.path.join(result_path, 'trueseg_putback.nii.gz'),)
    
    