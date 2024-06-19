import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def init_bn(bn_struct):
    nn.init.constant_(bn_struct.weight, 1.0)
    nn.init.constant_(bn_struct.bias, 0.0)


def init_struct(nn_struct):
    if not isinstance(nn_struct, nn.Module):
        return KeyError("Only nn.Module can be initialized!")
    for m in nn_struct.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def init_conv(layer):
    if not (isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d)):
        return KeyError("Only nn.Conv1d or nn.Conv2d can be initialized!")

    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


def init_conv2d(layer):
    if not isinstance(layer, nn.Conv2d):
        return KeyError("Only nn.Conv2d can be initialized!")

    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    # nn.init.xavier_uniform_(layer.weight, gain=1)       #20200421_3
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


def init_fc(layer):
    if not isinstance(layer, nn.Linear):
        return KeyError("Only nn.Linear can be initialized!")
    nn.init.normal_(layer.weight, 0, 0.01)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
