import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models import vgg


class TransformNetwork(nn.Module):

    def __init__(self):
        super(TransformNetwork, self).__init__()
        # initial layers
        self.conv_1 = ConvPadded(3, 32, kernel_size=9, stride=1)
        self.instance_norm_1 = nn.InstanceNorm2d(32, affine=True)
        self.conv_2 = ConvPadded(32, 64, kernel_size=3, stride=2)
        self.instance_norm_2 = nn.InstanceNorm2d(64, affine=True)
        self.conv_3 = ConvPadded(64, 128, kernel_size=3, stride=2)
        self.instance_norm_3 = nn.InstanceNorm2d(128, affine=True)
        # residual layers
        self.residual_1 = Residual(128)
        self.residual_2 = Residual(128)
        self.residual_3 = Residual(128)
        self.residual_4 = Residual(128)
        self.residual_5 = Residual(128)
        # upsampling layers
        self.deconv_1 = UpsampleConv(128, 64, kernel_size=3, stride=1, upsample=2)
        self.instance_norm_4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv_2 = UpsampleConv(64, 32, kernel_size=3, stride=1, upsample=2)
        self.instance_norm_5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv_3 = ConvPadded(32, 3, kernel_size=9, stride=1)

        # non linear layer
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.instance_norm_1(self.conv_1(x)))
        out = self.relu(self.instance_norm_2(self.conv_2(out)))
        out = self.relu(self.instance_norm_3(self.conv_3(out)))

        out = self.residual_1(out)
        out = self.residual_2(out)
        out = self.residual_3(out)
        out = self.residual_4(out)
        out = self.residual_5(out)

        out = self.relu(self.instance_norm_4(self.deconv_1(out)))
        out = self.relu(self.instance_norm_5(self.deconv_2(out)))
        out = self.deconv_3(out)

        return out


class ConvPadded(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvPadded, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Residual(nn.Module):

    def __init__(self, channels):
        super(Residual, self).__init__()
        self.conv_1 = ConvPadded(channels, channels, kernel_size=3, stride=1)
        self.instance_norm_1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv_2 = ConvPadded(channels, channels, kernel_size=3, stride=1)
        self.instance_norm_2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.instance_norm_1(self.conv_1(x)))
        out = self.instance_norm_2(self.conv_2(out))
        return out + x


class UpsampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample):
        super(UpsampleConv, self).__init__()
        self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.upsample_layer(x)
        out = self.reflection_pad(out)
        return self.conv2d(out)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class TransferVgg(nn.Module):

    def __init__(self, layers):
        super(TransferVgg, self).__init__()
        self.layer_names = []
        for layer_name, layer in layers:
            setattr(self, layer_name, layer)
            self.layer_names.append(layer_name)

    def forward(self, x, out_keys):
        out = {}
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            if isinstance(layer, nn.Conv2d):
                out[layer_name] = F.relu(layer(x))
            else:
                out[layer_name] = layer(x)
            x = out[layer_name]
        return {key: value for key, value in out.items() if key in out_keys}

    @classmethod
    def factory(cls, cfg, last_layer, mean, std):
        layers = [('normalization', Normalization(mean, std))]
        in_channels = 3
        conv_layer = 1
        conv_layer_number = 1
        pool_layer_number = 1
        for v in cfg:
            if v == 'M':
                layers.append(('pool_%s' % pool_layer_number, nn.MaxPool2d(kernel_size=2, stride=2)))
                pool_layer_number += 1
                conv_layer += 1
                conv_layer_number = 1
            else:
                conv_layer_name = 'conv_%s_%s' % (conv_layer, conv_layer_number)
                layers.append((conv_layer_name, nn.Conv2d(in_channels,
                                                          v,
                                                          kernel_size=3,
                                                          padding=1)))
                in_channels = v
                if conv_layer_name == last_layer:
                    break
                conv_layer_number += 1
        return cls(layers)


def transfer_vgg19(last_layer, device):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        model = TransferVgg.factory(cfg=vgg.cfg['E'],
                                    last_layer=last_layer,
                                    mean=mean,
                                    std=std)

        state_dict = model_zoo.load_url(vgg.model_urls['vgg19'])

        mapping = {
            'conv_1_1': 'features.0',
            'conv_1_2': 'features.2',
            'conv_2_1': 'features.5',
            'conv_2_2': 'features.7',
            'conv_3_1': 'features.10',
            'conv_3_2': 'features.12',
            'conv_3_3': 'features.14',
            'conv_3_4': 'features.16',
            'conv_4_1': 'features.19',
            'conv_4_2': 'features.21',
            'conv_4_3': 'features.23',
            'conv_4_4': 'features.25',
            'conv_5_1': 'features.28',
            'conv_5_2': 'features.30',
            'conv_5_3': 'features.32',
            'conv_5_4': 'features.34',
        }

        new_state_dict = {}
        for key, map_to in mapping.items():
            new_state_dict['%s.weight' % key] = state_dict['%s.weight' % map_to]
            new_state_dict['%s.bias' % key] = state_dict['%s.bias' % map_to]
            if key == last_layer:
                break
        model.load_state_dict(new_state_dict)
        for param in model.parameters():
            param.requires_grad = False
        return model.to(device).eval()