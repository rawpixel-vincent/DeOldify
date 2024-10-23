import argparse
import mmcv
from colorization_inference import colorization_inference, init_colorization_model
from mmedit.core import tensor2img


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_colorization_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    output = colorization_inference(model, args.img, device=args.device)
    output = tensor2img(output)

    # show the results
    if args.show:
        mmcv.imshow(output, 'predicted colorization result')

    # save
    if isinstance(args.out, str):
        mmcv.imwrite(output, args.out)


if __name__ == '__main__':
    args = dict([
        ("config",'configs/deoldify_stable_configs.py'),
        ("checkpoint",'checkpoints/ColorizeStable_gen.pth'),
        ("img",'b.jpg'),
        ("device",'cpu'),
        ("show",True),
        ("out",'d.jpg'),
    ])
    print(args)
    main(argparse.Namespace(**args))
