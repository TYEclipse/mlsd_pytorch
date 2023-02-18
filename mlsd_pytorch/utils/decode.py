import torch
import torch.nn as nn
import torch.nn.functional as F


def deccode_lines_TP(tpMap, score_thresh=0.1, len_thresh=2, topk_n=1000, ksize=3):
    '''
    tpMap:
        center: tpMap[1, 0, :, :]
        displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert b == 1, 'only support bsize==1'
    # 位置偏移
    displacement = tpMap[:, 1:5, :, :]

    # 中心点激活图
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)

    # 去掉邻域内非最大值
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep

    # 找出置信度最高的K个点
    heat = heat.reshape(-1, )
    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)

    # 找出置信度大于阈值的点
    valid_inx = torch.where(scores > score_thresh)
    scores = scores[valid_inx]
    indices = indices[valid_inx]

    # 计算中心点坐标
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)

    # 计算线段端点坐标
    center_ptss = torch.cat((xx, yy), dim=-1)
    start_point = center_ptss + displacement[0, :2, yy, xx].reshape(2, -1).permute(1, 0)
    end_point = center_ptss + displacement[0, 2:, yy, xx].reshape(2, -1).permute(1, 0)

    # 找出长度大于阈值的线段
    lines = torch.cat((start_point, end_point), dim=-1)
    lines_swap = torch.cat((end_point, start_point), dim=-1)
    all_lens = (end_point - start_point) ** 2
    all_lens = all_lens.sum(dim=-1)
    all_lens = torch.sqrt(all_lens)
    valid_inx = torch.where(all_lens > len_thresh)

    # 输出结果
    center_ptss = center_ptss[valid_inx]
    lines = lines[valid_inx]
    lines_swap = lines_swap[valid_inx]
    scores = scores[valid_inx]
    return center_ptss, lines, lines_swap, scores
