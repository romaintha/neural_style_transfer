def get_last_used_conv_layer(layer_names):
    layers = []
    for name in layer_names:
        a, b = name.split('_')[1:]
        layers.append((int(a), int(b)))
    return 'conv_%s_%s' % sorted(layers)[-1]