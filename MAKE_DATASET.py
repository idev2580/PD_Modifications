import os
import GET_PER_PIXEL_SE as gppe
from typing import List, Dict

DEFAULT_GT_SPP = 32768

def get_img_scene_name(imgfile_name:str):
    return imgfile_name.split('spp.')[0].split('_')[0]

def get_img_type(imgfile_name:str):
    return ".".join(imgfile_name.split('.')[1:])

def get_img_gt_file(imgfile_name:str, remove_type:bool=True, spp_level=DEFAULT_GT_SPP):
    return "{}_{:06}spp.{}".format(
        get_img_scene_name(imgfile_name),
        spp_level,
        'exr' if remove_type else get_img_type(imgfile_name)
    )

def make_dataset(dirname:str, is_test=False):
    den_dirname = os.path.join(dirname, "DENOISED/")
    err_gt_dir = os.path.join(dirname, "ERROR_GT/")
    sure_dir = os.path.join(dirname, "ORIG_SURE/")
    pprdsure_dir = os.path.join(dirname, "PPRDSURE/")
    gt_dir = os.path.join(dirname, "GT/")

    file_candidates = os.listdir(den_dirname)
    targets:List[str] = []
    for f in file_candidates:
        filedir = os.path.join(den_dirname, f)
        if(os.path.isfile(filedir)):
            targets.append(f)
    
    for f in targets:
        print(f)
        gt_file = get_img_gt_file(f)
        sure_file = f.split('.')[0] + ".rt_hdr_alb_nrm_var.sure.exr"
        #PPSE
        gppe.get_per_pixel_se(
            os.path.join(den_dirname, f),
            os.path.join(gt_dir, gt_file),
            os.path.join(err_gt_dir, f+".ppse.exr")
        )
        #PPRSE
        gppe.get_per_pixel_rse(
            os.path.join(den_dirname, f),
            os.path.join(gt_dir, gt_file),
            os.path.join(err_gt_dir, f+".pprse.exr")
        )
        #PPRDSURE - No theoretical reasons, and no practical benefits -> abandoned
        '''gppe.get_per_pixel_rdsure(
            os.path.join(den_dirname, f),
            os.path.join(gt_dir, gt_file),
            os.path.join(sure_dir, sure_file),
            os.path.join(pprdsure_dir, f+".pprdsure.exr")
        )'''
        
    return

if __name__ == '__main__':
    make_dataset('./DATASET/TRAIN/')
    make_dataset('./DATASET/VALID/')
    make_dataset('./DATASET/TEST/', True)
