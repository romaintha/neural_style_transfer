import argparse

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from image_handler import ImageHandler

from models import transfer_vgg19
from losses import ContentLoss, StyleLoss, VariationLoss
from utils import get_last_used_conv_layer


def run_style_transfer(image_size,
                       content_image_path,
                       style_image_path,
                       content_layers_weights,
                       style_layers_weights,
                       variation_weight,
                       n_steps,
                       shifting_activation_value,
                       device_name,
                       preserve_colors):
    print('Transfer style to content image')
    print('Number of iterations: %s' % n_steps)
    print('Preserve colors: %s' % preserve_colors)
    print('--------------------------------')
    print('Content image path: %s' % content_image_path)
    print('Style image path: %s' % style_image_path)
    print('--------------------------------')
    print('Content layers: %s' % content_layers_weights.keys())
    print('Content weight: %s' % style_layers_weights.keys())
    print('Style layers: %s' % content_layers_weights.values())
    print('Style weight: %s' % style_layers_weights.values())
    print('Variation weight: %s' % variation_weight)
    print('--------------------------------')
    print('Shifting activation value: %s' % shifting_activation_value)
    print('--------------------------------\n\n')

    device = torch.device("cuda" if (torch.cuda.is_available() and device_name == 'cuda') else "cpu")

    image_handler = ImageHandler(image_size=image_size,
                                 content_image_path=content_image_path,
                                 style_image_path=style_image_path,
                                 device=device,
                                 preserve_colors=preserve_colors)
    content_layer_names = list(content_layers_weights.keys())
    style_layer_names = list(style_layers_weights.keys())
    layer_names = content_layer_names + style_layer_names

    last_layer = get_last_used_conv_layer(layer_names)
    model = transfer_vgg19(last_layer, device)

    print('--------------------------------')
    print('Model:')
    print(model)
    print('--------------------------------')
    content_features = model(image_handler.content_image, content_layer_names)
    content_losses = {layer_name: ContentLoss(content_features[layer_name], weight=weight)
                      for layer_name, weight in content_layers_weights.items()}

    style_features = model(image_handler.style_image, style_layer_names)
    style_losses = {layer_name: StyleLoss(style_features[layer_name],
                                          weight=weight,
                                          shifting_activation_value=shifting_activation_value)
                    for layer_name, weight in style_layers_weights.items()}

    variation_loss = VariationLoss(weight=variation_weight)

    combination_image = image_handler.content_image.clone()
    optimizer = optim.LBFGS([combination_image.requires_grad_()])
    run = [0]
    while run[0] <= n_steps:
        def closure():
            # correct the values of updated input image
            combination_image.data.clamp_(0, 1)

            optimizer.zero_grad()
            out = model(combination_image, layer_names)
            variation_score = variation_loss(combination_image)
            content_score = torch.sum(torch.stack([loss(out[layer_name])
                                                   for layer_name, loss in content_losses.items()]))
            style_score = torch.sum(torch.stack([loss(out[layer_name])
                                                 for layer_name, loss in style_losses.items()]))

            loss = style_score + content_score + variation_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} Variation Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), variation_score.item()))
                print()

            return loss
        optimizer.step(closure)

        # a last correction...
    combination_image.data.clamp_(0, 1)

    plt.figure()
    image_handler.imshow(combination_image, title='Output Image')
    plt.show()
    return image_handler.image_unloader(combination_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('content_image_path',
                        help='the path to the content image')
    parser.add_argument('style_image_path',
                        help='the path to the style image')
    parser.add_argument('--combination_image_path',
                        default='',
                        help='the path to the combination image')
    parser.add_argument('--image_size',
                        type=int,
                        default=400,
                        help='the expected image size of the combination image')
    parser.add_argument('--content_layers',
                        default='conv_4_2',
                        help='a comma separated list of layers to use for content loss')
    parser.add_argument('--style_layers',
                        default='conv_1_1,conv_2_1,conv_3_1,conv_4_1,conv_5_1',
                        help='a comma separated list of layers to use for style loss')
    parser.add_argument('--content_weights',
                        type=str,
                        default='1',
                        help='''a comma separated list of weights to apply to content losses. If only one weight is passed, 
                        it will be apply to all layers. Otherwise the number of weights passed must matched the number 
                        of content layers''')
    parser.add_argument('--style_weights',
                        type=str,
                        default='64000,128000,256000,512000,512000',
                        help='''a comma separated list of weights to apply to style losses. If only one weight is passed, 
                        it will be apply to all layers. Otherwise the number of weights passed must matched the number 
                        of style layers''')
    parser.add_argument('--variation_weight',
                        type=float,
                        default=0.0001,
                        help='the weight to apply to variation loss')
    parser.add_argument('--n_steps',
                        type=int,
                        default=300,
                        help='the number of steps runs by the optimizer')
    parser.add_argument('--shifting_activation_value',
                        type=int,
                        default=0,
                        help='the activation value shift used in the calculation of the Gram matrix')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='the device on which to perform the training')
    parser.add_argument('--preserve_colors',
                        action='store_const',
                        const=True,
                        default=False,
                        help='whether to preserve the color of the content image during style transfer')

    args = parser.parse_args()

    content_layers = args.content_layers.split(',')
    style_layers = args.style_layers.split(',')

    content_weights = [float(w) for w in args.content_weights.split(',')]
    if len(content_weights) == 1:
        content_weights = [content_weights[0]] * len(content_layers)
    else:
        assert len(content_weights) == len(content_layers)
    content_layers_weights = {n: w for n, w in zip(content_layers, content_weights)}

    style_weights = [float(w) for w in args.style_weights.split(',')]
    if len(style_weights) == 1:
        style_weights = [style_weights[0]] * len(style_layers)
    else:
        assert len(style_weights) == len(style_layers)
    style_layers_weights = {n: w for n, w in zip(style_layers, style_weights)}

    combination_image = run_style_transfer(args.image_size,
                                           args.content_image_path,
                                           args.style_image_path,
                                           content_layers_weights,
                                           style_layers_weights,
                                           args.variation_weight,
                                           args.n_steps,
                                           args.shifting_activation_value,
                                           args.device,
                                           args.preserve_colors)
    if args.combination_image_path:
        combination_image.save(args.combination_image_path)