import json
import tqdm
import os
import cv2
import torch
import json
import random
import pickle
import numpy as np
from torch.utils.data import Dataset

from albumentations import (
    RandomBrightnessContrast,
    OneOf,
    HueSaturationValue,
    Compose,
    Normalize
)

from mlsd_pytorch.data.utils import \
    (swap_line_pt_maybe,
     get_ext_lines,
     gen_TP_mask2,
     gen_SOL_map,
     gen_junction_and_line_mask,
     TP_map_to_line_numpy,
     cut_line_by_xmin,
     cut_line_by_xmax)


def parse_label_file_info(img_dir, label_file):
    infos = []
    contens = json.load(open(label_file, 'r'))
    for c in tqdm.tqdm(contens):
        w = c['width']
        h = c['height']
        lines = c['lines']
        # fn = c['filename'][:-4]+'.jpg'
        fn = c['filename']
        full_fn = img_dir + fn
        assert os.path.exists(full_fn), full_fn

        json_content = {
            'version': '4.5.6',
            'flags': {},
            'shapes': [],
            'imagePath': fn,
            'imageData': None,
            'imageHeight': h,
            'imageWidth': w,
        }
        for l in lines:
            item = {
                "label": "line",
                "points": [
                    [
                        np.clip(np.float_(l[0]), 0, w),
                        np.clip(np.float_(l[1]), 0, h)
                    ],
                    [
                        np.clip(np.float_(l[2]), 0, w),
                        np.clip(np.float_(l[3]), 0, h)
                    ]
                ],
                "group_id": None,
                "shape_type": "line",
                "flags": {}
            }
            json_content['shapes'].append(item)
        infos.append(json_content)
    return infos


class Line_Dataset(Dataset):
    def __init__(self, cfg, is_train):
        super(Line_Dataset, self).__init__()

        self.with_centermap_extend = cfg.datasets.with_centermap_extend
        self.min_len = cfg.decode.len_thresh
        self.is_train = is_train

        if is_train:
            self.img_dir = cfg.train.img_dir
            self.label_fn = cfg.train.label_fn
        else:
            self.img_dir = cfg.val.img_dir
            self.label_fn = cfg.val.label_fn

        self.cache_dir = cfg.train.data_cache_dir
        self.with_cache = cfg.train.with_cache

        print("==> load label..")
        if self.with_cache:
            ann_cache_fn = self.cache_dir+"/" + \
                os.path.basename(self.label_fn)+".cache"
            if os.path.exists(ann_cache_fn):
                print("==> load {} from cache dir..".format(ann_cache_fn))
                self.anns = pickle.load(open(ann_cache_fn, 'rb'))
            else:
                self.anns = self._load_anns(self.img_dir, self.label_fn)
                print("==> cache to  {}".format(ann_cache_fn))
                pickle.dump(self.anns, open(ann_cache_fn, 'wb'))
        else:
            self.anns = self._load_anns(self.img_dir, self.label_fn)
        # random.shuffle(self.anns)
        print("==> valid samples: ", len(self.anns))

        self.input_size = cfg.datasets.input_size
        self.train_aug = self._aug_train()
        self.test_aug = self._aug_test()

        self.cache_to_mem = cfg.train.cache_to_mem
        self.cache_dict = {}
        if self.with_cache:
            print("===> cache...")
            for ann in tqdm.tqdm(self.anns):
                self.load_label(ann, False)

    def __len__(self):
        return len(self.anns)

    def _aug_train(self):
        aug = Compose(
            [
                OneOf(
                    [
                        HueSaturationValue(hue_shift_limit=10,
                                           sat_shift_limit=10,
                                           val_shift_limit=10,
                                           p=0.5),
                        RandomBrightnessContrast(brightness_limit=0.2,
                                                 contrast_limit=0.2,
                                                 p=0.5)
                    ]
                ),
            ],
            p=1.0)
        return aug

    def _aug_test(self):
        aug = Compose(
            [
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ],
            p=1.0)
        return aug

    def _line_len_fn(self, l1):
        len1 = np.sqrt((l1[2] - l1[0]) ** 2 + (l1[3] - l1[1]) ** 2)
        return len1

    def _load_anns(self, img_dir, label_fn):
        infos = parse_label_file_info(img_dir, label_fn)
        anns = []
        for c in infos:

            img_full_fn = os.path.join(img_dir, c['imagePath'])
            if not os.path.exists(img_full_fn):
                print(" not exist!".format(img_full_fn))
                exit(0)

            lines = []
            for s in c['shapes']:
                pt = s["points"]
                line = [pt[0][0], pt[0][1], pt[1][0], pt[1][1]]
                line = swap_line_pt_maybe(line)
                #line = np.array(line, np.float32)

                if not self.is_train:
                    lines.append(line)
                elif self._line_len_fn(line) > self.min_len:
                    lines.append(line)

            dst_ann = {
                'img_full_fn': img_full_fn,
                'lines': lines,
                'img_w': c['imageWidth'],
                'img_h': c['imageHeight']
            }
            anns.append(dst_ann)

        return anns

    def _crop_aug(self, img, ann_origin):
        assert img.shape[1] == ann_origin['img_w']
        assert img.shape[0] == ann_origin['img_h']
        img_w = ann_origin['img_w']
        img_h = ann_origin['img_h']
        lines = ann_origin['lines']
        xmin = random.randint(1, int(0.1 * img_w))

        #ymin = random.randint(1, 0.1 * img_h)
        #ymax = img_h - random.randint(1, 0.1 * img_h)

        # xmin
        xmin_lines = []
        for line in lines:
            flg, line = cut_line_by_xmin(line, xmin)
            line[0] -= xmin
            line[2] -= xmin
            if flg and self._line_len_fn(line) > self.min_len:
                xmin_lines.append(line)
        lines = xmin_lines

        img = img[:, xmin:, :]
        # xmax
        xmax = img.shape[1] - random.randint(1, int(0.1 * img.shape[1]))
        img = img[:, :xmax, :].copy()
        xmax_lines = []
        for line in lines:
            flg, line = cut_line_by_xmax(line, xmax)
            if flg and self._line_len_fn(line) > self.min_len:
                xmax_lines.append(line)
        lines = xmax_lines

        ann_origin['lines'] = lines
        ann_origin['img_w'] = img.shape[1]
        ann_origin['img_h'] = img.shape[0]

        return img, ann_origin

    def _geo_aug(self, img, ann_origin):
        do_aug = False

        lines = ann_origin['lines'].copy()
        if random.random() < 0.5:
            do_aug = True
            flipped_lines = []
            img = np.fliplr(img)
            for l in lines:
                flipped_lines.append(
                    swap_line_pt_maybe([ann_origin['img_w'] - l[0],
                                        l[1], ann_origin['img_w'] - l[2], l[3]]))
            ann_origin['lines'] = flipped_lines

        lines = ann_origin['lines'].copy()
        if random.random() < 0.5:
            do_aug = True
            flipped_lines = []
            img = np.flipud(img)
            for l in lines:
                flipped_lines.append(
                    swap_line_pt_maybe([l[0],
                                        ann_origin['img_h'] - l[1],
                                        l[2],
                                        ann_origin['img_h'] - l[3]]))
            ann_origin['lines'] = flipped_lines

        lines = ann_origin['lines'].copy()
        if random.random() < 0.5:
            do_aug = True
            r_lines = []
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            for l in lines:
                r_lines.append(
                    swap_line_pt_maybe([ann_origin['img_h'] - l[1],
                                        l[0], ann_origin['img_h'] - l[3], l[2]]))
            ann_origin['lines'] = r_lines
            ann_origin['img_w'] = img.shape[1]
            ann_origin['img_h'] = img.shape[0]

        if random.random() < 0.5:
            do_aug = True
            img, ann_origin = self._crop_aug(img, ann_origin)

        ann_origin['img_w'] = img.shape[1]
        ann_origin['img_h'] = img.shape[0]

        return do_aug, img, ann_origin

    def load_label(self, ann, do_aug):
        norm_lines = []
        for line in ann['lines']:
            norm_line = [
                np.clip(line[0] / ann['img_w'], 0, 1),
                np.clip(line[1] / ann['img_h'], 0, 1),
                np.clip(line[2] / ann['img_w'], 0, 1),
                np.clip(line[3] / ann['img_h'], 0, 1)
            ]

            assert norm_line[0] != norm_line[2] or norm_line[1] != norm_line[
                3], f'fatal err! {ann["img_w"]} {ann["img_h"]} {norm_line} {line} {ann}'

            norm_lines.append(norm_line)

        ann['norm_lines'] = norm_lines

        ann['norm_min_len'] = self.min_len / \
            np.sqrt(ann['img_w'] ** 2 + ann['img_h'] ** 2)

        label_cache_path = os.path.basename(ann['img_full_fn'])[:-4] + '.npy'
        label_cache_path = self.cache_dir + '/' + label_cache_path

        can_load = self.with_cache and not do_aug

        if can_load and self.cache_to_mem and label_cache_path in self.cache_dict.keys():
            label = self.cache_dict[label_cache_path]

        elif can_load and os.path.exists(label_cache_path):
            label = np.load(label_cache_path)
            if self.cache_to_mem:
                self.cache_dict[label_cache_path] = label

        else:
            tp_mask = gen_TP_mask2(ann['norm_lines'], self.input_size // 2, self.input_size // 2,
                                   with_ext=self.with_centermap_extend)
            sol_mask, _ = gen_SOL_map(ann['norm_lines'], self.input_size // 2, self.input_size // 2, ann['norm_min_len'],
                                      with_ext=False)
            junction_map, line_map = gen_junction_and_line_mask(ann['norm_lines'],
                                                                self.input_size // 2, self.input_size // 2)

            label = np.zeros((2 * 7 + 2, self.input_size // 2,
                             self.input_size // 2), dtype=np.float32)
            label[0:7, :, :] = sol_mask
            label[7:14, :, :] = tp_mask
            label[14, :, :] = junction_map[0]
            label[15, :, :] = line_map[0]

            if not do_aug and self.with_cache:
                #
                if self.cache_to_mem:
                    #print("cache to mem: {} [ total: {} ]".format(label_cache_path,len(self.cache_dict) ))
                    self.cache_dict[label_cache_path] = label
                else:
                    #print("cache to cache dir:", label_cache_path)
                    np.save(label_cache_path, label)

        return label

    def __getitem__(self, index):

        ann = self.anns[index].copy()
        image_origin = cv2.imread(ann['img_full_fn'])

        do_aug = False
        if self.is_train and random.random() < 0.5:
            do_aug, image_origin, ann = self._geo_aug(image_origin, ann)

        image_origin = cv2.resize(
            image_origin, (self.input_size, self.input_size))

        image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)

        label = self.load_label(ann, do_aug)

        norm_lines = ann['norm_lines']
        ext_lines = get_ext_lines(norm_lines, ann['norm_min_len'])

        tp_lines_list = norm_lines * self.input_size

        sol_lines_list = ext_lines * self.input_size

        tp_lines_tensor = torch.from_numpy(
            np.array(tp_lines_list, np.float32))
        sol_lines_tensor = torch.from_numpy(
            np.array(sol_lines_list, np.float32))

        if self.is_train:
            image_origin = self.train_aug(image=image_origin)['image']
        image = self.test_aug(image=image_origin)['image']

        return image, image_origin, label, \
            tp_lines_list, \
            tp_lines_tensor, \
            sol_lines_tensor, \
            ann['img_full_fn']


def LineDataset_collate_fn(batch):
    batch_size = len(batch)
    h, w, c = batch[0][0].shape

    images = np.zeros((batch_size, 3, h, w), dtype=np.float32)
    labels = np.zeros((batch_size, 16, h // 2, w // 2), dtype=np.float32)
    image_filename_list = []
    image_origin_list = []
    tp_lines_list_all = []
    tp_lines_tensor_all = []
    sol_lines_tensor_all = []

    for idx in range(batch_size):
        image, image_origin, label, \
            tp_lines_list, tp_lines_tensor, \
            sol_lines_tensor, img_fn = batch[idx]

        images[idx] = image.transpose((2, 0, 1))
        labels[idx] = label
        image_filename_list.append(img_fn)
        image_origin_list.append(image_origin)
        tp_lines_list_all.append(tp_lines_list)
        tp_lines_tensor_all.append(tp_lines_tensor)
        sol_lines_tensor_all.append(sol_lines_tensor)

    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)

    return {
        "images": images,
        "labels": labels,
        "image_filename_list": image_filename_list,
        "image_origin_list": image_origin_list,
        "gt_lines_list_all": tp_lines_list_all,
        "tp_lines_tensor_all": tp_lines_tensor_all,
        "sol_lines_tensor_all": sol_lines_tensor_all
    }


if __name__ == '__main__':
    from mlsd_pytorch.cfg.default import get_cfg_defaults

    cfg = get_cfg_defaults()

    root_dir = "/home/lhw/m2_disk/data/czcv_2021/wireframe_raw/"
    cfg.train.img_dir = root_dir + "/images/"
    cfg.train.label_fn = root_dir + "/valid.json"
    cfg.train.batch_size = 1
    cfg.train.data_cache_dir = "/home/lhw/m2_disk/data/czcv_2021/wireframe_cache/"
    cfg.train.with_cache = True
    cfg.datasets.with_centermap_extend = False

    dset = Line_Dataset(cfg, True)
    for img_norm, img, label, norm_lines_512, \
            norm_lines_512_tensor, sol_lines_512, fn in dset:
        # continue
        print(img.shape)
        print(label.shape)
        print(norm_lines_512_tensor.shape)
        centermap = label[7]
        centermap = centermap[np.newaxis, :]
        displacement_map = label[8:12]
        reverse_lines = TP_map_to_line_numpy(centermap, displacement_map)
        print("reverse_lines: ", len(reverse_lines))
        for i, l in enumerate(reverse_lines):
            #color = (random.randint(100, 255), random.randint(100, 255), 255)
            color = (0, 0, 255)
            x0, y0, x1, y1 = l
            cv2.line(img, (int(round(x0)), int(round(y0))),
                     (int(round(x1)), int(round(y1))), color, 1, 16)

        cv2.imwrite(root_dir + "/gui/gui_lines.jpg", img)
        cv2.imwrite(root_dir + "/gui/gui_centermap.jpg", centermap[0] * 255)

        displacement_map = displacement_map[0]
        displacement_map = np.where(displacement_map != 0, 255, 0)
        cv2.imwrite(root_dir + "/gui/gui_dis0.jpg", displacement_map)

        len_map = np.where(label[12] != 0, 255, 0)
        cv2.imwrite(root_dir + "/gui/gui_lenmap.jpg", len_map)

        centermap = label[0]
        centermap = centermap[np.newaxis, :]
        displacement_map = label[1:5]
        reverse_lines = TP_map_to_line_numpy(centermap, displacement_map)
        print("SOL reverse_lines: ", len(reverse_lines))
        for i, l in enumerate(reverse_lines):
            color = (random.randint(100, 255), random.randint(100, 255), 255)
            x0, y0, x1, y1 = l
            cv2.line(img, (int(round(x0)), int(round(y0))),
                     (int(round(x1)), int(round(y1))), color, 1, 16)

        cv2.imwrite(root_dir + "/gui/gui_SOL_lines.jpg", img)
        cv2.imwrite(root_dir + "/gui/gui_SOL_centermap.jpg",
                    centermap[0] * 255)

        displacement_map = displacement_map[0]
        displacement_map = np.where(displacement_map != 0, 255, 0)
        cv2.imwrite(root_dir + "/gui/gui_SOL_dis0.jpg", displacement_map)

        cv2.imwrite(root_dir + "/gui/gui_line_seg.jpg", label[0][15] * 255)
        cv2.imwrite(root_dir + "/gui/gui_junc_seg.jpg", label[0][14] * 255)

        break
