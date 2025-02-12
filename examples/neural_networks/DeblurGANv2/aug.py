from typing import List
import albumentations as albu


def get_transforms(size: int, scope: str = 'geometric', crop: str = 'random'):
    augs = {
        'weak'     : albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
            ]
        ),
        'geometric': albu.OneOf(
            [
                albu.HorizontalFlip(p=1.0),
                # 替换 ShiftScaleRotate 为 Affine 以消除警告
                albu.Affine(
                    rotate=(-45, 45),  # 随机旋转范围
                    scale=(0.9, 1.1),  # 随机缩放范围
                    translate_percent=(-0.0625, 0.0625),  # 随机平移比例（左右/上下均为 -6.25%~6.25%）
                    p=1.0
                ),
                albu.Transpose(p=1.0),
                albu.OpticalDistortion(p=1.0),
                albu.ElasticTransform(p=1.0),
            ]
        )
    }
    
    aug_fn = augs[scope]
    crop_fn = {
        'random': albu.RandomCrop(height=size, width=size, p=1.0),
        'center': albu.CenterCrop(height=size, width=size, p=1.0)
    }[crop]
    
    # 使用最新 API 的 PadIfNeeded（min_height 和 min_width）
    pad = albu.PadIfNeeded(min_height=size, min_width=size)
    
    pipeline = albu.Compose(
        [aug_fn, pad, crop_fn],
        additional_targets={'target': 'image'}
    )
    
    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']
    
    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose(
        [normalize],
        additional_targets={'target': 'image'}
    )
    
    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']
    
    return process


def _resolve_aug_fn(name):
    d = {
        'cutout'             : albu.CoarseDropout,
        'rgb_shift'          : albu.RGBShift,
        'hsv_shift'          : albu.HueSaturationValue,
        'motion_blur'        : albu.MotionBlur,
        'median_blur'        : albu.MedianBlur,
        'snow'               : albu.RandomSnow,
        'shadow'             : albu.RandomShadow,
        'fog'                : albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma'              : albu.RandomGamma,
        'sun_flare'          : albu.RandomSunFlare,
        'sharpen'            : albu.Sharpen,
        'jpeg'               : albu.ImageCompression,
        'gray'               : albu.ToGray,
        'pixelize'           : albu.Downscale,
        # ToDo: partial gray
    }
    return d[name]


def get_corrupt_function(config: List[dict]):
    augs = []
    for aug_params in config:
        name = aug_params.pop('name')
        if name == 'cutout':
            # 移除旧配置中不再被支持的参数，避免传入 CoarseDropout
            aug_params.pop('num_holes', None)
            aug_params.pop('max_h_size', None)
            aug_params.pop('max_w_size', None)
        # 对于 JPEG，不要转换 quality_lower/quality_upper 为 quality，
        # 直接使用配置中传入的参数（新版本 ImageCompression 直接使用 quality_lower 和 quality_upper）
        cls = _resolve_aug_fn(name)
        prob = aug_params.pop('prob', 0.5)
        augs.append(cls(p=prob, **aug_params))
    
    augs = albu.OneOf(augs)
    
    def process(x):
        return augs(image=x)['image']
    
    return process
