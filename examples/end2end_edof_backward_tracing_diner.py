import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from metrics import crop_and_evaluate_views

sys.path.append("../")

import diffoptics as do
from utils_end2end import load_deblurganv2, ImageFolder

import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --------------------------------------------------------------------------
# 创建镜头 + 准备
# --------------------------------------------------------------------------
lens = do.Lensgroup(device=device)
lens.load_file(Path('./lenses/end2end/end2end_edof_2.txt'))  # TODO: norminal design
lens.plot_setup2D()
[surface.to(device) for surface in lens.surfaces]

downsample_factor = 4
pixel_size = downsample_factor * 3.45e-3  # [mm]
film_size = [512 // downsample_factor, 512 // downsample_factor]
lens.prepare_mts(pixel_size, film_size)

print("========================================================================")
print('Check your lens:')
print(lens)
print("========================================================================")

wavelengths = [656.2725, 587.5618, 486.1327]  # (R,G,B)


def create_screen(texture: torch.Tensor, z: float, pixelsize: float) -> do.Screen:
    texturesize = np.array(texture.shape[0:2])
    screen = do.Screen(do.Transformation(np.eye(3), np.array([0, 0, z])), texturesize * pixelsize, texture, device=device)
    return screen


def render_single(wavelength: float, screen: do.Screen, sample_ray_function, images: list[torch.Tensor]):
    valid, ray_new = sample_ray_function(wavelength)
    uv, valid_screen = screen.intersect(ray_new)[1:]
    mask = valid & valid_screen
    
    I_batch = []
    for image in images:
        screen.update_texture(image[..., wavelengths.index(wavelength)])
        I_batch.append(screen.shading(uv, mask))
    
    return torch.stack(I_batch, axis=0), mask


def render(screen: do.Screen, images: list[torch.Tensor], ray_counts_per_pixel: int) -> torch.Tensor:
    Is = []
    for wavelength in wavelengths:
        I = 0
        M = 0
        for i in range(ray_counts_per_pixel):
            I_current, mask = render_single(wavelength, screen, lambda x: lens.sample_ray_sensor(x), images)
            I = I + I_current
            M = M + mask
        
        I = I / (M[None, ...] + 1e-10)
        I = I.reshape((len(images), *np.flip(np.asarray(film_size)))).permute(0, 2, 1)
        Is.append(I)
    return torch.stack(Is, axis=-1)


focal_length = 102


def render_gt(screen: do.Screen, images: list[torch.Tensor]) -> torch.Tensor:
    Is = []
    for wavelength in wavelengths:
        I, mask = render_single(wavelength, screen, lambda x: lens.sample_ray_sensor_pinhole(x, focal_length), images)
        I = I.reshape((len(images), *np.flip(np.asarray(film_size)))).permute(0, 2, 1)
        Is.append(I)
    return torch.stack(Is, axis=-1)


# --------------------------------------------------------------------------
# DINER
# --------------------------------------------------------------------------
from inr.diner import DINER, DINER_XY_polynomial

# ============================== Aspheric ==============================
# mask = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
mask = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # TODO
diner = DINER_XY_polynomial(
    in_features=1, out_features=1,
    hidden_features=32, hidden_layers=2,
    hash_table_length=10,
    out_mask=mask,
    first_omega_0=30., hidden_omega_0=6.
).to(dtype=torch.float32).to(device)
# ======================================================================

# ==============================   Mesh   ==============================
# W, H = 128, 128
# diner = DINER(
#     in_features=1, out_features=1,
#     hidden_features=64, hidden_layers=2,
#     hash_table_length=W * H,
#     first_omega_0=30., hidden_omega_0=6.
# ).to(dtype=torch.float32).to(device)
# ======================================================================

diner_optim = torch.optim.Adam(diner.parameters(), lr=1e-4)

# --------------------------------------------------------------------------
# 初始化网络 (DeblurGANv2)
# --------------------------------------------------------------------------
net = load_deblurganv2()
net.prepare()

print('Initial lens parameter (still 10 dims, but from MLP now).')

train_path = './training_dataset/'
train_dataloader = torch.utils.data.DataLoader(ImageFolder(train_path), batch_size=1, shuffle=False)
it = iter(train_dataloader)
image = next(it).squeeze().to(device)

settings = {
    'spp_forward'          : 100,
    'spp_backward'         : 20,
    'num_passes'           : 5,
    'image_batch_size'     : 5,
    'network_training_iter': 200,
    'num_of_training'      : 10,
    'savefig'              : True
}

if settings['savefig']:
    from pathlib import Path
    import os
    
    # opath = Path('end2end_output') / (str(datetime.now().strftime("%Y%m%d%H%M%S")) + "-DINER" + f"-mesh{W}*{H}")
    opath = Path('end2end_output') / (str(datetime.now().strftime("%Y%m%d%H%M%S")) + "-DINER" + f"{sum(mask)}")
    os.makedirs(opath, exist_ok=True)
    
    results_file = opath / 'training_log.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("z(mm)\tlens.surfaces[0].ai\tPSNR\tSSIM\tLPIPS\n")
        # f.write("z(mm)\tPSNR\tSSIM\tLPIPS\n")

# --------------------------------------------------------------------------
# wrapper_func: 用于 vjp
# --------------------------------------------------------------------------
def wrapper_func(screen, images, params):
    """
    params: shape=[N], 此处即 DINER 输出的参数.
    """
    # ============================== Aspheric ==============================
    # lens.surfaces[0].ai = params
    # ======================================================================
    
    # ==============================   Mesh   ==============================
    lens.surfaces[0].c = params
    # ======================================================================
    
    return render(screen, images, settings['spp_forward'])

# --------------------------------------------------------------------------
# 训练循环
# --------------------------------------------------------------------------
zs = [8e3, 6e3, 4.5e3]
pixelsizes = [0.1 * z / 6e3 for z in zs]

print('Training starts ...')
for iteration in range(settings['num_of_training']):
    for z_idx, z in enumerate(zs):
        
        print('============================================================')
        print(f'Iteration = {iteration}, z = {z} mm')
        print('============================================================')
        
        screen = create_screen(image, z, pixelsizes[z_idx])
        
        # (1) 前向渲染
        tq = tqdm(range(settings['image_batch_size']))
        tq.set_description('(1) Rendering batch images')
        
        images = []
        for image_idx in tq:
            try:
                data = next(it)
            except StopIteration:
                it = iter(train_dataloader)
                data = next(it)
            image = data.squeeze().to(device)
            images.append(image.clone())
        tq.close()
        with torch.no_grad():
            # ============================== Aspheric ==============================
            # diner_output = diner(None)["model_out"].squeeze(-1)
            # lens.surfaces[0].ai = diner_output
            # ======================================================================
            
            # ==============================   Mesh   ==============================
            diner_output = diner(None)["model_out"].squeeze(-1).reshape(W, H)
            lens.surfaces[0].c = diner_output
            # ======================================================================
            
            Is = render(screen, images, settings['spp_forward'])
            Is_gt = render_gt(screen, images)
        
        # 可视化： reshape + 归一化
        Is_view = np.concatenate([I.cpu().numpy().astype(np.uint8) for I in Is], axis=1)
        Is_gt_view = np.concatenate([I.cpu().numpy().astype(np.uint8) for I in Is_gt], axis=1)
        Is = 2 * torch.permute(Is, (0, 3, 1, 2)) / 255 - 1
        Is_gt = 2 * torch.permute(Is_gt, (0, 3, 1, 2)) / 255 - 1
        
        # (2) 训练网络
        Is_output = net.run(Is, Is_gt, is_inference=False, num_iters=settings['network_training_iter'], desc='(2) Training net weights')
        Is_output_np = np.transpose(255 / 2 * (Is_output.detach().cpu().numpy() + 1), (0, 2, 3, 1)).astype(np.uint8)
        Is_output_view = np.concatenate([I for I in Is_output_np], axis=1)
        
        # 裁剪 + 评估
        cropped_Is_view, cropped_Is_gt_view, cropped_Is_output_view, psnr_val, ssim_val, lpips_val = crop_and_evaluate_views(Is_view, Is_gt_view, Is_output_view)
        print('PSNR={:.2f}, SSIM={:.4f}, LPIPS={:.4f}'.format(psnr_val, ssim_val, lpips_val))
        # ============================== Aspheric ==============================
        # print('lens.surfaces[0].ai:', lens.surfaces[0].ai)
        # ======================================================================
        
        # 保存可视化
        if settings['savefig']:
            import matplotlib.pyplot as plt
            
            fig, axs = plt.subplots(3, 1)
            for idx, (imgv, title) in enumerate(zip([cropped_Is_view, cropped_Is_gt_view, cropped_Is_output_view], ['Input', 'GT', 'Network Output'])):
                axs[idx].imshow(imgv)
                axs[idx].set_title(title)
                axs[idx].axis('off')
            fig.tight_layout()
            fig.savefig(str(opath / f'iter_{iteration}_z={z}mm_images.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # (3) 再一次 forward + get gradient wrt Is
        Is.requires_grad = True
        Is_output = net.run(Is, Is_gt, is_inference=False, num_iters=1)
        Is_grad = Is.grad.permute(0, 2, 3, 1)
        del Is, Is_gt, Is_output
        torch.cuda.empty_cache()
        
        # (4) 多次 vjp 累加
        # 这里直接使用 DINER 输出作为叶子变量
        # ============================== Aspheric ==============================
        # param_nd = diner(None)["model_out"].squeeze(-1).detach().clone()
        # ======================================================================
        
        # ==============================   Mesh   ==============================
        param_nd = diner(None)["model_out"].squeeze(-1).reshape(W, H).detach().clone()
        # ======================================================================
        param_nd.requires_grad = True  # 让它成为leaf param, 方便手动更新
        
        dthetas = torch.zeros_like(param_nd)
        tq = tqdm(range(settings['num_passes']))
        tq.set_description('(3) Back-prop optical parameters')
        
        for inner_iter in tq:  # 这里vjp: f(param_nd)-> image  => back with Is_grad
            vjp_val = torch.autograd.functional.vjp(lambda p: wrapper_func(screen, images, p), param_nd, Is_grad)[1]
            dthetas += vjp_val
        
        # (5) 根据element-wise lr更新 param_nd
        # with torch.no_grad():
        #     param_nd -= manual_lr * dthetas
        
        # (6) 将 vjp 累计梯度传回 DINER
        diner_optim.zero_grad()
        # ============================== Aspheric ==============================
        #updated_diner_output = diner(None)["model_out"].squeeze(-1)
        # ======================================================================
        
        # ==============================   Mesh   ==============================
        updated_diner_output = diner(None)["model_out"].squeeze(-1).reshape(W, H)
        # ======================================================================
        
        # (6-1) 计算 "delta" = 新param - 旧param
        delta = param_nd.detach() - updated_diner_output  # shape=[10]
        
        # (6-2) backward
        # 注意: updated_diner_output 是一段graph => 反传会到 diner
        updated_diner_output.backward(gradient=delta)
        
        # (7) diner_optim.step()
        diner_optim.step()
        
        # 将本轮结果写入 txt 文件
        # ============================== Aspheric ==============================
        # current_ai_list = lens.surfaces[0].ai.detach().cpu().tolist()
        # ai_str = "[" + ", ".join(f"{x:e}" for x in current_ai_list) + "]"
        # with open(results_file, 'a', encoding='utf-8') as f:
        #     f.write(f"{z}\t{ai_str}\t{psnr_val:.2f}\t{ssim_val:.4f}\t{lpips_val:.4f}\n")
        # ======================================================================
        
        # ==============================   Mesh   ==============================
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"{z}\t{psnr_val:.2f}\t{ssim_val:.4f}\t{lpips_val:.4f}\n")
        # ======================================================================
