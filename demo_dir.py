import torch
import os
import sys
import cv2
import numpy as np

# from mlsd_pytorch.models.mbv2_mlsd import MobileV2_MLSD
from mlsd_pytorch.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from mlsd_pytorch.cfg.default import get_cfg_defaults

from utils import pred_lines

import argparse

IMAGE_SIZE = 1280

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='mlsd_pytorch/configs/mobilev2_mlsd_large_1280_base2_bsize2_LSDdataset_only_link.yaml')
    parser.add_argument("--model_path", type=str, default='workdir/models/mobilev2_mlsd_large_1280_base2_bsize2_LSDdataset_only_link/best.pth')
    parser.add_argument("--img_dir", type=str, default='data/LSD_dataset/images')
    parser.add_argument("--output_path", type=str, default='output')
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--len_thresh", type=int, default=5)
    return parser.parse_args()

def main():
    args = get_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    current_dir = os.path.dirname(__file__)
    if current_dir == "":
        current_dir = "./"
    # model_path = current_dir+'/models/mlsd_tiny_512_fp32.pth'
    # model = MobileV2_MLSD_Tiny().cuda().eval()

    model_path = os.path.join(current_dir, args.model_path)
    model = MobileV2_MLSD_Large(cfg).cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        model_path, map_location=device), strict=True)
    
    img_file_list = os.listdir(args.img_dir)
    img_suffix = ['jpg', 'png', 'JPG', 'PNG']
    img_file_list = [x for x in img_file_list if x.split('.')[-1] in img_suffix]

    for img_file in img_file_list:
        img_fn = os.path.join(current_dir, args.img_dir, img_file)

        img = cv2.imread(img_fn)
        ori_size = img.shape[:2]
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # lines, scores = pred_lines(img, model, [1280, 1280], args.score_thresh, args.len_thresh)
        lines, scores = pred_lines(img, model, [IMAGE_SIZE, IMAGE_SIZE], args.score_thresh, args.len_thresh)
        
        # trans to original size
        lines[:, 0::2] = lines[:, 0::2] * ori_size[1] / IMAGE_SIZE
        lines[:, 1::2] = lines[:, 1::2] * ori_size[0] / IMAGE_SIZE

        # output lines to txt file
        file_name = img_file.split('.')[0]
        output_path = os.path.join(current_dir, args.output_path, '{}_out.txt'.format(file_name))
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        # sort lines by score increasing
        line_list = [{'line': l, 'score': s} for l, s in zip(lines, scores)]
        line_list.sort(key=lambda x: x['score'])
        lines = [x['line'] for x in line_list]
        scores = [x['score'] for x in line_list]

        with open(output_path, 'w') as f:
            for l, score in zip(lines, scores):
                f.write('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(l[0].item(), l[1].item(), l[2].item(), l[3].item(), score))

        img2 = cv2.imread(img_fn)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        for l, score in zip(lines, scores):
            # score -> color, 0->(255, 0, 0), 1->(0, 0, 255)
            color = (int(255 * (1 - score)), 0, int(255 * score))
            cv2.line(img2, (int(l[0]), int(l[1])),
                    (int(l[2]), int(l[3])), color, 1, cv2.LINE_AA)
            # print score near the center of the line
            cv2.putText(img2, '{:.4f}'.format(score), (int((l[0] + l[2]) / 2), int((l[1] + l[3]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        output_path = os.path.join(current_dir, args.output_path, '{}_out.jpg'.format(file_name))
        cv2.imwrite(output_path, img2)


if __name__ == '__main__':
    main()
