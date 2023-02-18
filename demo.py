import argparse
import os
import sys

import cv2
import torch

from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from utils import pred_lines


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='mlsd_pytorch/configs/mobilev2_mlsd_large_512_base2_bsize24_LSDdataset_only_link.yaml')
    parser.add_argument("--model_path", type=str, default='workdir/models/mobilev2_mlsd_large_512_base2_bsize24_LSDdataset_only_link/best.pth')
    parser.add_argument("--img_path", type=str, default='data/LSD_dataset/images/幻灯片402.PNG')
    parser.add_argument("--output_path", type=str, default='output')
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

    img_fn = os.path.join(current_dir, args.img_path)

    img = cv2.imread(img_fn)
    img = cv2.resize(img, (512, 512))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    score_thresh = 0.1
    lines, scores = pred_lines(img, model, [512, 512], score_thresh, 10)
    
    # output lines to txt file
    output_path = os.path.join(current_dir, args.output_path, 'frame_out.txt')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for l, score in zip(lines, scores):
            f.write('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(l[0].item(), l[1].item(), l[2].item(), l[3].item(), score))


    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for l in lines:
        cv2.line(img, (int(l[0]), int(l[1])),
                 (int(l[2]), int(l[3])), (0, 0, 256), 1, cv2.LINE_AA)
    output_path = os.path.join(current_dir, args.output_path, 'frame_out.jpg')
    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    main()
