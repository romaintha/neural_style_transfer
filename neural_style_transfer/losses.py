import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, weight):
        super(ContentLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        return self.weight * F.mse_loss(input, target)


class StyleLoss(nn.Module):

    def __init__(self, weight, shifting_activation_value=0):
        super(StyleLoss, self).__init__()
        self.shifting_activation_value = shifting_activation_value
        self.weight = weight

    def forward(self, input, target):
        G_input = self.gram_matrix(input)
        G_target = self.gram_matrix(target).detach()
        return self.weight * F.mse_loss(G_input, G_target)

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d) + self.shifting_activation_value
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class VariationLoss(nn.Module):

    def __init__(self, weight):
        super(VariationLoss, self).__init__()
        self.weight = weight

    def forward(self, input):
        batch_size, n_channels, image_height, image_width = input.size()
        a = input[:, :, :image_height-1, :image_width-1] - input[:, :, 1:, :image_width-1]

        b = input[:, :, :image_height-1, :image_width-1] - input[:, :, :image_height-1, 1:]
        return self.weight * torch.sum((a**2 + b**2).pow(1.25))