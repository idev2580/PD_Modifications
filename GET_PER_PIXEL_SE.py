import OpenEXR
import sys
import numpy as np

#https://pypi.org/project/OpenEXR/

def get_per_pixel_se(target_filename:str,
                     truth_filename:str,
                     output_dir:str = ""):
    with OpenEXR.File(target_filename) as tgfile:
        with OpenEXR.File(truth_filename) as gtfile:
            tg_rgb = tgfile.channels()["RGB"].pixels
            gt_rgb = gtfile.channels()["RGB"].pixels

            tg_h, tg_w, tg_channel = tg_rgb.shape
            gt_h, gt_w, gt_channel = gt_rgb.shape

            #Check if they have same shape
            if(tg_h != gt_h) or (tg_w != gt_w):
                print("Shape differs, abort.")
                return False
            #print("Width={}, Height={}, Channel={}".format(tg_w, tg_h, tg_channel))

            diff_rgb = gt_rgb - tg_rgb
            sq_diff_rgb = np.square(diff_rgb)
            sq_diff_header = {
                "compression" : OpenEXR.ZIP_COMPRESSION,
                "type" : OpenEXR.scanlineimage
            }
            sq_diff_channels = {"RGB" : sq_diff_rgb}

            
            pixel_se_filename = ".".join(target_filename.split('.')[:-1]) + '.ppse.exr' if output_dir == "" else output_dir

            with OpenEXR.File(sq_diff_header, sq_diff_channels) as psfile:
                psfile.write(pixel_se_filename)
            return True
        return False
    return False

def get_per_pixel_rdsure(
    ori_filename:str,
    den_filename:str,
    sure_filename:str,
    output_dir:str = ""
):
    with OpenEXR.File(ori_filename) as ofile, OpenEXR.File(den_filename) as dfile, OpenEXR.File(sure_filename) as sfile:
        o_rgb = ofile.channels()["RGB"].pixels
        d_rgb = dfile.channels()["RGB"].pixels
        s_rgb = sfile.channels()["RGB"].pixels

        #Max RGB
        m_rgb = np.maximum(o_rgb, d_rgb)
        pprdsure_rgb = np.true_divide(
            np.square(s_rgb),
            np.add(
                np.square(m_rgb),
                1.0 / 100.0
            )
        )
        pprdsure_header = {
            "compression" : OpenEXR.ZIP_COMPRESSION,
            "type" : OpenEXR.scanlineimage
        }
        pprdsure_channels = {"RGB" : pprdsure_rgb}
        pprdsure_filename = ".".join(target_filename.split('.')[:-1]) + '.pprdsure.exr' if output_dir == "" else output_dir

        with OpenEXR.File(pprdsure_header, pprdsure_channels) as pprdsurefile:
            pprdsurefile.write(pprdsure_filename)
        return True
    return False
    

def get_per_pixel_rse(target_filename:str,
                     truth_filename:str,
                     output_dir:str = ""):
    with OpenEXR.File(target_filename) as tgfile:
        with OpenEXR.File(truth_filename) as gtfile:
            tg_rgb = tgfile.channels()["RGB"].pixels
            gt_rgb = gtfile.channels()["RGB"].pixels

            tg_h, tg_w, tg_channel = tg_rgb.shape
            gt_h, gt_w, gt_channel = gt_rgb.shape

            #Check if they have same shape
            if(tg_h != gt_h) or (tg_w != gt_w):
                print("Shape differs, abort.")
                return False

            diff_rgb = gt_rgb - tg_rgb
            r_sq_diff_rgb = np.true_divide(
                np.square(diff_rgb),
                np.add(
                    np.square(gt_rgb),
                    1.0 / 100.0
                )
            )
            r_sq_diff_header = {
                "compression" : OpenEXR.ZIP_COMPRESSION,
                "type" : OpenEXR.scanlineimage
            }
            r_sq_diff_channels = {"RGB" : r_sq_diff_rgb}

            pixel_rse_filename = ".".join(target_filename.split('.')[:-1]) + '.pprse.exr' if output_dir == "" else output_dir

            with OpenEXR.File(r_sq_diff_header, r_sq_diff_channels) as prsfile:
                prsfile.write(pixel_rse_filename)
            return True
        return False
    return False

def sum_of_img(target_filename:str):
    with OpenEXR.File(target_filename) as tfile:
        retval = (np.mean(tfile.channels()["RGB"].pixels))
    return retval

if __name__ == "__main__":
    if(len(sys.argv) == 3):
        target_filename = sys.argv[1]
        truth_filename = sys.argv[2]
        get_per_pixel_se(target_filename, truth_filename)
        get_per_pixel_rse(target_filename, truth_filename)
    else:
        print("Only 2 arguments(target, truth) must be given")
