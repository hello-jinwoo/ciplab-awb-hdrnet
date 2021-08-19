import os
import cv2
import numpy as np
import rawpy

def visualize(input_patch, pred_patch, gt_patch, data_root='data', templete='galaxy'):
    """
    Visualize model inference result.
    1. Re-bayerize RGB image by duplicating G pixels.
    2. Copy bayer pattern image into rawpy templete instance
    3. Use user_wb to render RGB image
    4. Crop proper size of patch from rendered RGB image
    """

    height, width, _ = input_patch.shape
    raw = rawpy.imread(os.path.join(data_root, templete + ".dng"))

    white_level = raw.white_level

    if templete == 'sony':
        black_level = 512
        white_level = raw.white_level / 4
    else:
        black_level = min(raw.black_level_per_channel)
        white_level = raw.white_level
        
    input_rgb = input_patch.astype('uint16')
#     output_rgb = pred_patch.astype('uint16')
    output_rgb = np.clip(pred_patch, 0, white_level).astype('uint16')
    gt_rgb = gt_patch.astype('uint16')

    input_bayer = bayerize(input_rgb, templete, black_level)
    output_bayer = bayerize(output_rgb, templete, black_level)
    gt_bayer = bayerize(gt_rgb, templete, black_level)

    input_rendered = render(raw, white_level, input_bayer, height, width, "maintain")
    awb_rendered = render(raw, white_level, input_bayer, height, width, "daylight_wb")
    output_rendered = render(raw, white_level, output_bayer, height, width, "maintain")
    gt_rendered = render(raw, white_level, gt_bayer, height, width, "maintain")

    return np.hstack([input_rendered, awb_rendered, output_rendered, gt_rendered])

def bayerize(img_rgb, camera, black_level):
    h,w,c = img_rgb.shape

    bayer_pattern = np.zeros((h*2,w*2))
    
    if camera == "galaxy":
        bayer_pattern[0::2,1::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,2] # B

    return bayer_pattern + black_level

def render(raw, white_level, bayer, height, width, wb_method):
    raw_mat = raw.raw_image
    for h in range(height*2):
        for w in range(width*2):
            raw_mat[h,w] = bayer[h,w]

    if wb_method == "maintain":
        user_wb = [1.,1.,1.,1.]
    elif wb_method == "daylight_wb":
        user_wb = raw.daylight_whitebalance

    rgb = raw.postprocess(user_sat=white_level, user_wb=user_wb, half_size=True, no_auto_bright=False)
    rgb_croped = rgb[0:height,0:width,:]
    
    return rgb_croped