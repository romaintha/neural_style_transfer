import argparse
import torch
from torch.optim import Adam

from models import TransformNetwork, transfer_vgg19

from losses import ContentLoss, StyleLoss, VariationLoss

from utils import get_last_used_conv_layer

from image_handler import TrainStyleImageHandler


def train(image_size,
          style_image_path,
          dataset_path,
          model_path,
          content_layers_weights,
          style_layers_weights,
          variation_weight,
          shifting_activation_value,
          batch_size,
          learning_rate,
          epochs,
          device_name):
    print('Train style model')
    print('Number of epochs: %s' % epochs)
    print('Leaning rate: %s' % learning_rate)
    print('Batch size: %s' % batch_size)
    print('--------------------------------')
    print('Style image path: %s' % style_image_path)
    print('Dataset path: %s' % dataset_path)
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

    content_layer_names = list(content_layers_weights.keys())
    style_layer_names = list(style_layers_weights.keys())
    layer_names = content_layer_names + style_layer_names

    last_layer = get_last_used_conv_layer(layer_names)
    vgg_model = transfer_vgg19(last_layer, device)

    image_handler = TrainStyleImageHandler(image_size, style_image_path, dataset_path, batch_size, device)

    transform_model = TransformNetwork().to(device)
    transform_model.train()
    optimizer = Adam(transform_model.parameters(), learning_rate)

    content_losses = {layer_name: ContentLoss(weight=weight)
                      for layer_name, weight in content_layers_weights.items()}
    style_losses = {layer_name: StyleLoss(weight=weight,
                                          shifting_activation_value=shifting_activation_value)
                    for layer_name, weight in style_layers_weights.items()}
    variation_loss = VariationLoss(weight=variation_weight)

    print('Start training')
    for epoch in range(epochs):
        print('--------------------')
        print('Epoch %s' % epoch)
        print('--------------------')
        epoch_content_score = 0
        epoch_style_score = 0
        epoch_variation_score = 0
        for batch_id, (batch, _) in enumerate(image_handler.train_loader):
            optimizer.zero_grad()
            transform_batch = transform_model(batch.to(device))

            transform_batch_features = vgg_model(transform_batch, layer_names)
            batch_features = vgg_model(batch.to(device), layer_names)
            style_features = vgg_model(image_handler.style_image, layer_names)
            variation_score = variation_loss(transform_batch)
            content_score = torch.sum(torch.stack([loss(transform_batch_features[layer_name],
                                                        batch_features[layer_name])
                                                   for layer_name, loss in content_losses.items()]))
            style_score = torch.sum(torch.stack([loss(transform_batch_features[layer_name],
                                                      style_features[layer_name])
                                                 for layer_name, loss in style_losses.items()]))

            loss = style_score + content_score + variation_score
            loss.backward()
            optimizer.step()

            epoch_content_score += content_score
            epoch_style_score += style_score
            epoch_variation_score += variation_score

            if batch_id % 50 == 0:
                print("Batch {}:".format(batch_id))
                print('Style Loss : {:4f} Content Loss: {:4f} Variation Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), variation_score.item()))

        print('Epoch summary \n Style Loss : {:4f} Content Loss: {:4f} Variation Loss: {:4f}'.format(
            epoch_style_score.item(), epoch_content_score.item(), epoch_variation_score.item()))

    transform_model.eval().cpu()
    torch.save(transform_model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('style_image_path',
                        help='the path to the style image')
    parser.add_argument('dataset_path',
                        help='the path to the dataset folder')
    parser.add_argument('model_path',
                        help='the path where to save the model')
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
    parser.add_argument('--shifting_activation_value',
                        type=int,
                        default=0,
                        help='the activation value shift used in the calculation of the Gram matrix')
    parser.add_argument('--batch_size',
                        type=int,
                        default=30,
                        help='the batch size used during training')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='the learning_rate used during training')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='the number of epochs to trained')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='the device on which to perform the training')

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

    train(args.image_size,
          args.style_image_path,
          args.dataset_path,
          args.model_path,
          content_layers_weights,
          style_layers_weights,
          args.variation_weight,
          args.shifting_activation_value,
          args.batch_size,
          args.learning_rate,
          args.epochs,
          args.device)