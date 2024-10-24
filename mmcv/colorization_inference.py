import numpy as np
import torch
import PIL
from PIL import Image

import mmcv
from mmcv.parallel import collate, scatter
from mmedit.models import build_model
from mmedit.datasets.pipelines import Compose

import torchvision.transforms as transforms


def init_colorization_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.model.pretrained = None
    # config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)

    generator = model.generator
    if checkpoint is not None:
        # checkpoint = load_checkpoint(model, checkpoint)
        params = torch.load(checkpoint, map_location='cpu')

        # ------------generator keys transform------------------
        keys_0 = generator.state_dict().keys()
        # torch.save({'generator': generator.state_dict()}, 'keys.pth')

        keys_1 = params['model'].keys()
        # print(keys_0 == keys_1)

        if keys_0 != keys_1 and len(keys_0) == len(keys_1):
            d = params['model'].items()
            d1 = generator.state_dict().items()
            from collections import OrderedDict
            new_d = OrderedDict()
            for (k, v), (k1, v1) in zip(d, d1):
                new_d[k1] = v
            params['model'] = new_d
        # ------------------------------

        generator.load_state_dict(params['model'])

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def denorm(x, mean, std):
    x = x * std[..., None, None] + mean[..., None, None]
    return x


def overlay_color(generated_color_image, gray_image:Image.Image):
    color_y, color_u, color_v = generated_color_image.convert("YCbCr").split()
    orig_y, orig_u, orig_v = gray_image.convert("YCbCr").split()
    final = Image.merge("YCbCr", (orig_y, color_u, color_v)).convert("RGB")
    return final


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def post_process(result, extra_result, img_gray):
    results:torch.Tensor = result['img_color_fake'].squeeze(0)
    extra_results:torch.Tensor = extra_result['img_color_fake'].squeeze(0) if extra_result is not None else None

    # denom
    mean = torch.tensor([0.4850, 0.4560, 0.4060])  # imagenet的均值和方差
    std = torch.tensor([0.2290, 0.2240, 0.2250])
    results = denorm(results.detach(), mean, std)
    if extra_results is not None:
      mean = torch.tensor([0.4850, 0.4560, 0.4060])  # imagenet的均值和方差
      std = torch.tensor([0.2290, 0.2240, 0.2250])
      extra_results = denorm(extra_results.detach(), mean, std)

    # clamp
    results = results.float().clamp(min=0, max=1)
    extra_results = extra_results.float().clamp(min=0, max=1) if extra_results is not None else None


    # resize to results size
    extra_results = transforms.Resize(results.shape[-2:])(extra_results) if extra_results is not None else None









    # To PIL
    out:Image.Image = transforms.ToPILImage()(results)
    extra_out:Image.Image = transforms.ToPILImage()(extra_results) if extra_results is not None else None
    out = Image.blend(out, extra_out, 0.40) if extra_results is not None else out
    # img_gray = (img_gray.cpu().numpy()*255).astype('uint8').transpose(1, 2, 0)
    # img_gray = Image.fromarray(img_gray)

    # Resize
    orig_image:Image.Image = None
    if isinstance(img_gray, str):
        orig_image = PIL.Image.open(img_gray).convert('RGB')
    if isinstance(img_gray, np.ndarray):
        orig_image = Image.fromarray(np.uint8(img_gray))

    raw_color = out.resize(orig_image.size, resample=PIL.Image.LANCZOS)

    # overlay color
    final = overlay_color(raw_color, orig_image)

    # auto_contrast
    # final = mmcv.image.adjust_color(np.asarray(final), 1.1)
    # final = Image.fromarray(final)
    # final = mmcv.image.adjust_contrast(np.asarray(final), 1.2)
    # final = Image.fromarray(final)
    if extra_results is None:
      final = mmcv.image.adjust_contrast(np.asarray(final), 0.9)
      final = Image.fromarray(final)

    #normalize
    # final = normalize(np.asarray(final))
    # final = Image.fromarray(final.astype('uint8'), 'RGB')

    # # return final
    # if isinstance(img_gray, str):
    #     return final
    # if isinstance(img_gray, np.ndarray):
    #     return np.asarray(final)[:, :, ::-1]
    # return None
    if isinstance(img_gray, str):
        return transforms.ToTensor()(final)
    if isinstance(img_gray, np.ndarray):
        return np.asarray(final)[:, :, ::-1]
    return None


def colorization_inference(model, img, device='cuda:0', apply_extra=False):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        np.ndarray: The predicted colorization result.
    """

    # cfg = model.cfg
    # device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    if model.cfg.get('extra_pipeline', None):
        extra_pipeline = model.cfg.extra_pipeline

    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipelines in [test_pipeline, extra_pipeline]:
          for pipeline in list(pipelines):
              if 'key' in pipeline and key == pipeline['key']:
                  pipelines.remove(pipeline)
              if 'keys' in pipeline and key in pipeline['keys']:
                  pipeline['keys'].remove(key)
                  if len(pipeline['keys']) == 0:
                      pipelines.remove(pipeline)
              if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                  pipeline['meta_keys'].remove(key)

    # build the data pipeline
    test_pipeline = Compose(test_pipeline)
    extra_pipeline = Compose(extra_pipeline) if apply_extra else None

    # prepare data
    data = None
    extra_data = None
    if isinstance(img, str):
        data = dict(img_gray_path=img)
        if apply_extra:
          extra_data = dict(img_gray_path=img)
    if isinstance(img, np.ndarray):
        data = dict(img_gray=img)
        if apply_extra:
          extra_data = dict(img_gray=img)
    data = test_pipeline(data)
    extra_data = extra_pipeline(extra_data) if apply_extra else None
    if device == 'cpu':
      data = collate([data], samples_per_gpu=1)
      extra_data = collate([extra_data], samples_per_gpu=1) if apply_extra else None
    else:
      data = scatter(collate([data], samples_per_gpu=1), [device])[0]
      extra_data = scatter(collate([extra_data], samples_per_gpu=1), [device])[0] if apply_extra else None
    # # forward the model
    # model.eval()
    # with torch.no_grad():
    #     results = model.forward(data['gray_img']).squeeze()

    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)
        extra_result = model(test_mode=True, **extra_data) if apply_extra else None

    final = post_process(result, extra_result, img)

    return final
