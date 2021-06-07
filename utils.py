from geffnet.efficientnet_builder import decode_arch_def, resolve_bn_args
from geffnet.gen_efficientnet import _create_model
from torch import nn


def gen_efficientnet_lite_y(
    variant,
    in_c,
    feature_dim,
    channel_multiplier=1.0,
    depth_multiplier=1.0,
    pretrained=False,
    **kwargs
):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s2_e1_c16"],
        ["ir_r2_k3_s2_e2_c32"],
        ["ir_r2_k3_s2_e2_c64"],
        # ['ir_r1_k3_s2_e3_c80'],
        # ['ir_r3_k5_s1_e3_c112'],
        # ['ir_r4_k5_s2_e3_c192'],
        # ['ir_r1_k3_s1_e3_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=feature_dim,
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=nn.ReLU6,
        norm_kwargs=resolve_bn_args(kwargs),
        in_chans=in_c,
        **kwargs,
    )
    model = _create_model(model_kwargs, variant, pretrained)
    return model


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
