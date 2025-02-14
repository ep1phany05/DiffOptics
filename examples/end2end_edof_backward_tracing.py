import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from inr.diner import DINER

# 将上级目录加入搜索路径，保证可以导入 diffoptics 等模块
sys.path.append("../")

import diffoptics as do
from utils_end2end import dict_to_tensor, tensor_to_dict, load_deblurganv2, ImageFolder

import warnings

warnings.filterwarnings("ignore")

# 设定随机种子，保证结果可复现
torch.manual_seed(0)

# ------------------------------------------------------------------------------
# 初始化设备（CPU或GPU）
# ------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------------------------------------------------------------------------
# 创建镜头实例并加载光学参数
# ------------------------------------------------------------------------------
lens = do.Lensgroup(device=device)

# 加载镜头文件（包括镜片曲率、材料等信息）
lens.load_file(Path('./lenses/end2end/end2end_edof.txt'))  # norminal design

# 可视化镜头的2D光路图
lens.plot_setup2D()

# 将镜头所有面放到当前计算设备上
[surface.to(device) for surface in lens.surfaces]

# ------------------------------------------------------------------------------
# 设置传感器像素大小与分辨率
# ------------------------------------------------------------------------------
# ! downsample_factor: 降采样因子
downsample_factor = 4  # downsampled for run

# ! pixel_size: 单个像素的尺寸 (mm)
pixel_size = downsample_factor * 3.45e-3  # [mm]

# ! film_size: 感光面像素数量 (H, W)
film_size = [512 // downsample_factor, 512 // downsample_factor]

# 预处理镜头的光线追踪参数 (包括传感器尺寸、坐标系等)
lens.prepare_mts(pixel_size, film_size)

print("========================================================================")
print('Check your lens:')
print(lens)
print("========================================================================")

# ------------------------------------------------------------------------------
# 设置三种光谱对应的波长 (单位: nm)
# ------------------------------------------------------------------------------
wavelengths = [656.2725, 587.5618, 486.1327]  # (red, green, blue)


# ------------------------------------------------------------------------------
# 创建屏幕（用于放置场景纹理）
# ------------------------------------------------------------------------------
def create_screen(texture: torch.Tensor, z: float, pixelsize: float) -> do.Screen:
    """
    根据给定的纹理图像与位置 z 创建一个Screen对象。

    参数:
    texture: torch.Tensor, 图像纹理
    z: float, 屏幕放置的 z 位置 (mm)
    pixelsize: float, 屏幕像素大小 (mm)

    返回:
    screen: do.Screen, 封装了纹理与位姿信息的屏幕对象
    """
    texturesize = np.array(texture.shape[0:2])
    screen = do.Screen(
        do.Transformation(np.eye(3), np.array([0, 0, z])),
        texturesize * pixelsize,
        texture,
        device=device
    )
    return screen


# ------------------------------------------------------------------------------
# 渲染单波长
# ------------------------------------------------------------------------------
def render_single(wavelength: float, screen: do.Screen, sample_ray_function, images: list[torch.Tensor]):
    """
    使用指定的波长和射线采样函数对多个图像进行渲染。

    参数:
    wavelength: float, 当前渲染波长
    screen: do.Screen, 屏幕对象
    sample_ray_function: 函数, 生成射线的函数 (如 lens.sample_ray_sensor)
    images: list[torch.Tensor], 待渲染图像列表

    返回:
    I_batch: torch.Tensor, 渲染得到的图像批次 (按通道叠加)
    mask: torch.Tensor, 有效像素掩码
    """
    valid, ray_new = sample_ray_function(wavelength)
    # 计算屏幕交点 (uv坐标) 以及有效像素范围
    uv, valid_screen = screen.intersect(ray_new)[1:]
    mask = valid & valid_screen
    
    I_batch = []
    for image in images:
        # 更新屏幕纹理到当前波长通道
        screen.update_texture(image[..., wavelengths.index(wavelength)])
        # 利用屏幕的着色功能获取渲染结果
        I_batch.append(screen.shading(uv, mask))
    
    return torch.stack(I_batch, axis=0), mask


# ------------------------------------------------------------------------------
# 渲染主函数：多波长整合
# ------------------------------------------------------------------------------
def render(screen: do.Screen, images: list[torch.Tensor], ray_counts_per_pixel: int) -> torch.Tensor:
    """
    对给定屏幕和图像列表进行多波长渲染，累加并平均多次射线采样。

    参数:
    screen: do.Screen, 屏幕对象
    images: list[torch.Tensor], 图像列表
    ray_counts_per_pixel: int, 每个像素的射线采样数 (spp)

    返回:
    torch.Tensor, 维度 [batch, H, W, 3] 的RGB结果
    """
    Is = []
    for wavelength in wavelengths:
        # 初始化累加器
        I = 0
        M = 0
        for i in range(ray_counts_per_pixel):
            I_current, mask = render_single(
                wavelength,
                screen,
                lambda x: lens.sample_ray_sensor(x),
                images
            )
            I = I + I_current
            M = M + mask
        
        # 对有效像素进行平均
        I = I / (M[None, ...] + 1e-10)
        # 对渲染图进行维度变换 (batch, width, height)
        I = I.reshape((len(images), *np.flip(np.asarray(film_size)))).permute(0, 2, 1)
        Is.append(I)
    
    # 将多波长结果拼接 (batch, H, W, RGB)
    return torch.stack(Is, axis=-1)


# ------------------------------------------------------------------------------
# 通过针孔相机模型渲染“理想”GT图像
# ------------------------------------------------------------------------------
focal_length = 102  # [mm]


def render_gt(screen: do.Screen, images: list[torch.Tensor]) -> torch.Tensor:
    """
    使用针孔模型 (pinhole) 对场景进行成像，用作 Ground Truth (理想情况)。
    """
    Is = []
    for wavelength in wavelengths:
        I, mask = render_single(
            wavelength,
            screen,
            lambda x: lens.sample_ray_sensor_pinhole(x, focal_length),
            images
        )
        I = I.reshape((len(images), *np.flip(np.asarray(film_size)))).permute(0, 2, 1)
        Is.append(I)
    return torch.stack(Is, axis=-1)


# ------------------------------------------------------------------------------
# 优化光学表面参数示例 (可拓展为更复杂的曲面/模型)
# ------------------------------------------------------------------------------
# 下方注释掉的示例展示了可以在表面上优化多项式系数等操作
# XY_surface = (
#     a[0] +
#     a[1] * x + a[2] * y +
#     a[3] * x**2 + a[4] * x*y + a[5] * y**2 +
#     a[6] * x**3 + a[7] * x**2*y + a[8] * x*y**2 + a[9] * y**3
# )
#
# diner = DINER(
#             in_features=1, out_features=1,
#             hidden_features=32, hidden_layers=2,
#             hash_table_length=10,
#             first_omega_0=30, hidden_omega_0=6
#         ).to(dtype=torch.float32).to(device)
# lens.surfaces[0].ai = diner(None)["model_out"].squeeze(-1) * 1e-9 * torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).to(device)

diff_parameters = [lens.surfaces[0].ai]

# 指定学习率 (仅对后四个系数进行学习，其他为0)
# ! learning_rates: 针对每个差异化参数单独设置学习率
learning_rates = {'surfaces[0].ai': 1e-15 * torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).to(device)}

# 确保学习率与参数长度一致
for diff_para, key in zip(diff_parameters, learning_rates.keys()):
    if len(diff_para) != len(learning_rates[key]):
        raise Exception('Learning rates of {} is not of equal length to the parameters!'.format(key))
    diff_para.requires_grad = True

diff_parameter_labels = learning_rates.keys()

# ------------------------------------------------------------------------------
# 初始化网络 (DeblurGANv2 作为示例去卷积网络)
# ------------------------------------------------------------------------------
net = load_deblurganv2()
net.prepare()

print('Initial:')
current_parameters = [x.detach().cpu().numpy() for x in diff_parameters]
print('Current optical parameters are:')
for x, label in zip(current_parameters, diff_parameter_labels):
    print('-- lens.{}: {}'.format(label, x))

# ------------------------------------------------------------------------------
# 加载训练数据 (图像文件夹)
# ------------------------------------------------------------------------------
train_path = './training_dataset/'
train_dataloader = torch.utils.data.DataLoader(
    ImageFolder(train_path), batch_size=1, shuffle=False
)
it = iter(train_dataloader)
image = next(it).squeeze().to(device)

# ------------------------------------------------------------------------------
# 训练设置
# ------------------------------------------------------------------------------
# ! settings: 控制前向/后向采样数、训练迭代次数等
settings = {
    'spp_forward'          : 100,  # (Rays per pixel) 前向渲染射线采样数
    'spp_backward'         : 20,  # 后向传播时，每次的射线采样数
    'num_passes'           : 5,  # 累积后向传播的次数
    'image_batch_size'     : 5,  # 每批次训练的图像数
    'network_training_iter': 200,  # 网络训练迭代次数
    'num_of_training'      : 10,  # 外层训练循环次数
    'savefig'              : True  # 是否保存中间可视化结果
}

# 如果需要保存可视化结果，则创建对应输出目录
if settings['savefig']:
    opath = Path('end2end_output') / str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    opath.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 将渲染封装为一个函数，便于反向传播时的梯度计算 (VJP)
# ------------------------------------------------------------------------------
def wrapper_func(screen, images, squeezed_diff_parameters, diff_parameters, diff_parameter_labels):
    """
    将可学习参数替换到 lens 中，并进行 forward 渲染，用于后向传播计算梯度。
    """
    unpacked_diff_parameters = tensor_to_dict(squeezed_diff_parameters, diff_parameters)
    for idx, label in enumerate(diff_parameter_labels):
        exec('lens.{} = unpacked_diff_parameters[{}]'.format(label, idx))
    return render(screen, images, settings['spp_forward'])


# ------------------------------------------------------------------------------
# 设置屏幕位置和对应的像素大小 (模拟不同景深或景物距离)
# ------------------------------------------------------------------------------
zs = [8e3, 6e3, 4.5e3]  # [mm]
pixelsizes = [0.1 * z / 6e3 for z in zs]  # [mm]

print('Training starts ...')
for iteration in range(settings['num_of_training']):
    for z_idx, z in enumerate(zs):
        
        current_parameters = [x.detach().cpu().numpy() for x in diff_parameters]
        print('========================================================================')
        print('Iteration = {}, z = {} [mm]:'.format(iteration, z))
        print('Current optical parameters are:')
        for x, label in zip(current_parameters, diff_parameter_labels):
            print('-- lens.{}: {}'.format(label, x))
        print('========================================================================')
        
        # 创建屏幕 (调整 z 和 像素大小)
        screen = create_screen(image, z, pixelsizes[z_idx])
        
        # (1) 前向渲染一批图像
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
            # 渲染得到失焦图像和理想针孔GT图像
            Is = render(screen, images, settings['spp_forward'])
            Is_gt = render_gt(screen, images)
        
        # 可视化：拼接图像以便比较
        Is_view = np.concatenate([I.cpu().numpy().astype(np.uint8) for I in Is], axis=1)
        Is_gt_view = np.concatenate([I.cpu().numpy().astype(np.uint8) for I in Is_gt], axis=1)
        
        # 将维度从 (batch,H,W,RGB) 转为 (batch,RGB,H,W)，且像素归一化到 [-1,1]
        Is = 2 * torch.permute(Is, (0, 3, 1, 2)) / 255 - 1
        Is_gt = 2 * torch.permute(Is_gt, (0, 3, 1, 2)) / 255 - 1
        
        # (2) 训练网络权重
        Is_output = net.run(
            Is, Is_gt, is_inference=False,
            num_iters=settings['network_training_iter'],
            desc='(2) Training network weights'
        )
        Is_output_np = np.transpose(
            255 / 2 * (Is_output.detach().cpu().numpy() + 1), (0, 2, 3, 1)
        ).astype(np.uint8)
        Is_output_view = np.concatenate([I for I in Is_output_np], axis=1)
        del Is_output_np
        
        # 保存可视化结果
        if settings['savefig']:
            fig, axs = plt.subplots(3, 1)
            for idx, I_view, label in zip(
                    range(3),
                    [Is_view, Is_gt_view, Is_output_view],
                    ['Input', 'Ground truth', 'Network output']
            ):
                axs[idx].imshow(I_view)
                axs[idx].set_title(label + ' image(s)')
                axs[idx].set_axis_off()
            fig.tight_layout()
            fig.savefig(
                str(opath / 'iter_{}_z={}mm_images.png'.format(iteration, z)),
                dpi=400, bbox_inches='tight', pad_inches=0.1
            )
            fig.clear()
            plt.close(fig)
        
        # (3) 对输出图像进行再一次前向计算以获取梯度
        Is.requires_grad = True
        Is_output = net.run(Is, Is_gt, is_inference=False, num_iters=1)
        
        # 从网络输出计算损失梯度，回传给输入图像 Is
        Is_grad = Is.grad.permute(0, 2, 3, 1)
        del Is, Is_gt, Is_output
        torch.cuda.empty_cache()
        
        # 计算光学参数的梯度累积
        tq = tqdm(range(settings['num_passes']))
        tq.set_description('(3) Back-prop optical parameters')
        
        dthetas = torch.zeros_like(dict_to_tensor(diff_parameters)).detach()
        
        # 多次累加后向传播 (num_passes 次)
        for inner_iteration in tq:
            dthetas += torch.autograd.functional.vjp(
                lambda x: wrapper_func(screen, images, x, diff_parameters, diff_parameter_labels),
                dict_to_tensor(diff_parameters),
                Is_grad
            )[1]
        tq.close()
        
        # 更新光学参数 (镜头表面ai等)
        with torch.no_grad():
            for label, diff_para, dtheta in zip(
                    diff_parameter_labels,
                    diff_parameters,
                    tensor_to_dict(dthetas, diff_parameters)
            ):
                diff_para -= learning_rates[label] * dtheta.squeeze() / settings['num_passes']
                diff_para.grad = None
