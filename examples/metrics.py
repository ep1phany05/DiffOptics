import math
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import lpips  # pip install lpips

####################################
# 手写 SSIM 与 PSNR 函数实现
####################################

def gaussian(window_size, sigma):
    # 生成一维高斯核
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                            for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    # 生成二维高斯窗口，并扩展到指定通道数
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def SSIM(img1, img2):
    """
    计算两幅图像的 SSIM 值。
    输入:
      img1, img2: torch.Tensor, 形状 (N, C, H, W)，数值范围假定为 [0, 1]
    """
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def PSNR(img1, img2):
    """
    计算两幅图像的 PSNR 值。
    输入:
      img1, img2: numpy 数组，形状 (H, W, 3)，像素值归一化到 [0, 1]
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


####################################
# 裁剪和指标计算函数
####################################

def crop_single_view(view):
    """
    对单个拼接后的视图进行裁剪：
    输入 view: numpy 数组，形状 (128, 640, 3)，由 5 张 (128,128,3) 图像水平拼接而成。
    裁剪操作：对每个 128x128 的块，裁去上方 32 行和左侧 32 列，得到 (96,96,3) 的图像，
    最后水平拼接成形状 (96, 96*5, 3)。
    """
    cropped_blocks = []
    num_blocks = 5      # 每个视图由 5 张图片拼接而成
    block_width = 128
    crop_top, crop_left = 32, 32
    crop_h, crop_w = 96, 96  # 128 - 32 = 96

    for i in range(num_blocks):
        # 提取第 i 张图像块
        block = view[:, i * block_width:(i + 1) * block_width, :]
        # 裁剪：去掉上方 32 行和左侧 32 列
        cropped = block[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w, :]
        cropped_blocks.append(cropped)

    # 水平拼接 5 张 (96,96,3) 的图像
    cropped_view = np.concatenate(cropped_blocks, axis=1)  # 形状 (96, 96*5, 3)
    return cropped_view

def crop_and_evaluate_views(Is_view, Is_gt_view, Is_output_view):
    """
    对输入的三个拼接视图（形状均为 (128,640,3)）进行裁剪，
    使得每个视图变为 (96, 96*5, 3)。
    随后计算裁剪后 Is_output_view 与 Is_gt_view 之间的 PSNR、SSIM（手写版）和 LPIPS。

    参数:
      - Is_view:         numpy 数组，形状 (128,640,3)
      - Is_gt_view:      numpy 数组，形状 (128,640,3)
      - Is_output_view:  numpy 数组，形状 (128,640,3)

    返回:
      - cropped_Is_view, cropped_Is_gt_view, cropped_Is_output_view: 裁剪后的视图
      - psnr_val:  PSNR 值
      - ssim_val:  SSIM 值
      - lpips_val: LPIPS 距离（值越小越好）
    """
    # 对三个视图分别进行裁剪
    # cropped_Is_view = crop_single_view(Is_view)
    # cropped_Is_gt_view = crop_single_view(Is_gt_view)
    # cropped_Is_output_view = crop_single_view(Is_output_view)
    cropped_Is_view = Is_view
    cropped_Is_gt_view = Is_gt_view
    cropped_Is_output_view = Is_output_view

    # 使用自定义 PSNR 函数计算 PSNR（输入为 numpy 数组，数值范围为 [0,1]）
    psnr_val = PSNR(cropped_Is_gt_view / 255., cropped_Is_output_view / 255.)

    # 计算 SSIM：先将 numpy 数组转换为 torch.Tensor，形状 (1,3,H,W)
    gt_tensor = torch.from_numpy(cropped_Is_gt_view).permute(2, 0, 1).unsqueeze(0).float()
    output_tensor = torch.from_numpy(cropped_Is_output_view).permute(2, 0, 1).unsqueeze(0).float()
    ssim_val = SSIM(gt_tensor, output_tensor).item()

    # 计算 LPIPS：要求输入图像归一化到 [-1,1]
    gt_lpips = 2 * gt_tensor - 1
    output_lpips = 2 * output_tensor - 1
    loss_fn = lpips.LPIPS(net='alex')
    lpips_val = loss_fn(output_lpips, gt_lpips).item()

    return cropped_Is_view, cropped_Is_gt_view, cropped_Is_output_view, psnr_val, ssim_val, lpips_val


if __name__ == '__main__':
    # 模拟输入数据，这里生成随机图像，实际使用时替换为渲染结果
    Is_view = np.random.rand(128, 640, 3).astype(np.float32)
    Is_gt_view = np.random.rand(128, 640, 3).astype(np.float32)
    Is_output_view = np.random.rand(128, 640, 3).astype(np.float32)

    cropped_Is_view, cropped_Is_gt_view, cropped_Is_output_view, psnr_val, ssim_val, lpips_val = crop_and_evaluate_views(
        Is_view, Is_gt_view, Is_output_view)

    print("裁剪后 Is_view 的形状:", cropped_Is_view.shape)            # (96, 96*5, 3)
    print("裁剪后 Is_gt_view 的形状:", cropped_Is_gt_view.shape)          # (96, 96*5, 3)
    print("裁剪后 Is_output_view 的形状:", cropped_Is_output_view.shape)  # (96, 96*5, 3)
    print("PSNR:", psnr_val)
    print("SSIM:", ssim_val)
    print("LPIPS:", lpips_val)
