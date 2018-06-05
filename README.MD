# Neural Style Transfer
Implementation with Pytorch of neural style transfer

## Prerequisites
Install Pytorch (depending on your cuda version): https://pytorch.org/

Run:
```bash
pip install -r requirements.txt
```

## Style transfer
### Losses
It uses three kinds of losses:
* Content loss as described in [[1]](#references)
* Style loss as described in [[1]](#references)
* Variational loss

### Features
In addition to the style transfer described in [[1]](#references), some extra features are available:
* Preserving the colors of the content image as described in [[2]](#references).
* Shfiting activation values as described in [[3]](#references).

### Command
To perform a style transfer, run this command:
```bash
python neural_style_transfer/transfer.py -h
usage: transfer.py [-h] 
                   [--combination_image_path COMBINATION_IMAGE_PATH]
                   [--image_size IMAGE_SIZE] 
                   [--content_layers CONTENT_LAYERS]
                   [--style_layers STYLE_LAYERS]
                   [--content_weights CONTENT_WEIGHTS]
                   [--style_weights STYLE_WEIGHTS]
                   [--variation_weight VARIATION_WEIGHT] 
                   [--n_steps N_STEPS]
                   [--shifting_activation_value SHIFTING_ACTIVATION_VALUE]
                   [--device {cuda,cpu}]
                   [--preserve_colors]
                   content_image_path style_image_path

Neural Style Transfer

positional arguments:
  content_image_path    the path to the content image
  style_image_path      the path to the style image

optional arguments:
  -h, --help            show this help message and exit
  --combination_image_path COMBINATION_IMAGE_PATH
                        the path to the combination image
  --image_size IMAGE_SIZE
                        the expected image size of the combination image
  --content_layers CONTENT_LAYERS
                        a comma separated list of layers to use for content
                        loss
  --style_layers STYLE_LAYERS
                        a comma separated list of layers to use for style loss
  --content_weights CONTENT_WEIGHTS
                        a comma separated list of weights to apply to content
                        losses. If only one weight is passed, it will be apply
                        to all layers. Otherwise the number of weights passed
                        must matched the number of content layers
  --style_weights STYLE_WEIGHTS
                        a comma separated list of weights to apply to style
                        losses. If only one weight is passed, it will be apply
                        to all layers. Otherwise the number of weights passed
                        must matched the number of style layers
  --variation_weight VARIATION_WEIGHT
                        the weight to apply to variation loss
  --n_steps N_STEPS     the number of steps runs by the optimizer
  --shifting_activation_value SHIFTING_ACTIVATION_VALUE
                        the value by which the Gram matrix are shifted
  --device {cuda,cpu}   the device on which to perform the training
  --preserve_colors     whether to preserve the color of the content image
                        during style transfer

```

e.g.:
* Only specifying the content and style image paths:
```bash
python neural_style_transfer/transfer.py images/mountain.jpg images/monet.jpg
```

* If you wish to save the combination image:
```bash
python neural_style_transfer/transfer.py images/mountain.jpg images/monet.jpg  --combination_image combination_image.jpg
```

* If you wish to provide the content and style layers:
```bash
python neural_style_transfer/transfer.py images/mountain.jpg images/monet.jpg  --content_layers conv_4_2,conv_5_2 --style_layers conv_1_1,conv_2_1,conv_3_1,conv_4_1,conv_5_1
```

* If you wish to change the weights:
```bash
python neural_style_transfer/transfer.py images/mountain.jpg images/monet.jpg  --content_weights 1 --style_weights 64000,128000,256000,512000,512000
```

* Only specifying the content and style image paths:
```bash
python neural_style_transfer/transfer.py images/mountain.jpg images/monet.jpg  --preserve_colors
```

### Examples
* Using the default options:

Content: ![content_image](images/mountain.jpg =160x106) 
Style: ![style_image](images/monet.jpg =160x106)
Combination: ![combination_image](images/combination_image.jpg =125x100) 

* Preserving the colors:

Content: ![content_image](images/mountain.jpg =160x106) 
Style: ![style_image](images/monet.jpg =160x106)
Combination: ![combination_image](images/combination_image_colors_preserved.jpg =125x100) 

## References
*  [1]["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) Gatys et al
*  [2]["Preserving Color in Neural Artistic Style Transfer"](https://arxiv.org/abs/1606.05897) Gatys et al
*  [3]["Improving the Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1605.04603) Novak et al
