import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence

from utils import gen_efficientnet_lite_y
from utils import BasicBlock, conv1x1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Seq_Classifier_TE(nn.Module):
    def __init__(self, in_feature_dim, class_nums, max_len):
        super().__init__()
        self.max_len = max_len
        self.positional_encoding = PositionalEncoding(
            in_feature_dim, dropout=0, max_len=max_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature_dim, nhead=4, dim_feedforward=4 * 32, dropout=0
        )

        encoder_norm = nn.LayerNorm(in_feature_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, norm=encoder_norm
        )
        # ===========================
        self.decoder = nn.Sequential(
            nn.Linear(in_feature_dim, 128), nn.ReLU(True), nn.Linear(128, class_nums)
        )
        # self.decoder = nn.Linear(in_feature_dim, class_nums)
        self.sequence_fuse_maxlen = nn.Sequential(
            nn.Linear(max_len, 128), nn.ReLU(True), nn.Linear(128, 1)
        )
        # ===========================

        padding_mask_idx = gen_precalcu_padding_mask_idx(total_len=max_len)
        self.register_buffer("padding_mask_idx", padding_mask_idx)

    def forward(self, input):
        # x, lens_packed = self.pad_input_f(input)
        lens_pad = torch.LongTensor([len(v) for v in input])

        x = custom_padding_seq(input, self.max_len)

        padding_mask = gen_padding_mask(lens_pad, self.padding_mask_idx)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        x = x.permute(1, 2, 0)
        x = self.sequence_fuse_maxlen(x)
        x = x.squeeze()

        x = self.decoder(x)
        return x


class Seq_Classifier_CNN(nn.Module):
    def __init__(self, in_feature_dim, class_nums):
        super().__init__()
        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(
                512,
                512,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512, class_nums)

    def forward(self, x):
        x += 1
        pass
        return x


class Seq_Classifier_LSTM(nn.Module):
    def __init__(self, in_feature_dim, lstm_layer_num, lstm_dim_num, class_nums):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=in_feature_dim,
            hidden_size=lstm_dim_num,
            num_layers=lstm_layer_num,
        )

        self.decoder = nn.Sequential(
            nn.Linear(lstm_dim_num, 128), nn.ReLU(True), nn.Linear(128, class_nums)
        )

    def forward(self, input):
        in_len = set([len(x) for x in input])
        if len(in_len) != 1:
            x = pack_sequence(input, enforce_sorted=False)
            output, _ = self.rnn(x)
            seq_unpacked, lens_unpacked = pad_packed_sequence(output)
            lens_unpacked -= 1
            seq_out = torch.stack(
                [
                    seq_unpacked[lens_unpacked[i], i, :]
                    for i in range(len(lens_unpacked))
                ]
            )
            x = self.decoder(seq_out)
        else:
            x = torch.stack(input, dim=1)
            output, _ = self.rnn(x)
            seq_out = output[-1, ...]
            x = self.decoder(seq_out)
        return x


class RAW_Feature_Encoder_1dcnn(nn.Module):
    def __init__(self, in_c, out_feature_dim):
        super().__init__()
        self.m = gen_efficientnet_lite_y(None, in_c, out_feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.m.features(x)
        x = self.pool(x).squeeze()
        return x


class RAW_Feature_Encoder_2dcnn(nn.Module):
    def __init__(self, in_c, out_feature_dim):
        super().__init__()
        block = BasicBlock
        layers = [1, 1, 1, 1]
        self.inplanes = 64
        self.in_c = in_c
        self.conv1 = nn.Conv1d(
            in_c, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.final_l = nn.Linear(256, out_feature_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2).reshape(-1, self.in_c, 100)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).squeeze()
        x = self.final_l(x)

        return x


class TCA_FeatureEncoder(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(in_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_feature_dim),
        )

    def forward(self, input):
        return self.l(input)


class TCA_SeqEncoder_LSTM(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, lstm_layer_num, lstm_dim_num):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_feature_dim,
            hidden_size=lstm_dim_num,
            num_layers=lstm_layer_num,
        )
        self.decoder = nn.Sequential(
            nn.Linear(lstm_dim_num, 128), nn.ReLU(True), nn.Linear(128, out_feature_dim)
        )

    def forward(self, input):
        x = pack_sequence(input, enforce_sorted=False)
        output, _ = self.rnn(x)
        seq_unpacked, lens_unpacked = pad_packed_sequence(output)
        lens_unpacked -= 1
        seq_out = torch.stack(
            [seq_unpacked[lens_unpacked[i], i, :] for i in range(len(lens_unpacked))]
        )
        x = self.decoder(seq_out)

        return x


class TCA_SeqEncoder_TE(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, max_len):
        super().__init__()
        self.max_len = max_len

        self.positional_encoding = PositionalEncoding(
            in_feature_dim, dropout=0, max_len=max_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature_dim, nhead=2, dim_feedforward=64, dropout=0
        )

        encoder_norm = nn.LayerNorm(in_feature_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, norm=encoder_norm
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_feature_dim, 64), nn.ReLU(True), nn.Linear(64, out_feature_dim)
        )

        self.sequence_fuse_maxlen = nn.Sequential(
            nn.Linear(max_len, 64), nn.ReLU(True), nn.Linear(64, 1)
        )

        padding_mask_idx = gen_precalcu_padding_mask_idx(total_len=max_len)
        self.register_buffer("padding_mask_idx", padding_mask_idx)

    def forward(self, input):
        lens_pad = torch.LongTensor([len(v) for v in input])
        x = custom_padding_seq(input, self.max_len)
        padding_mask = gen_padding_mask(lens_pad, self.padding_mask_idx)

        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        x = x.permute(1, 2, 0)
        x = self.sequence_fuse_maxlen(x)
        x = x.squeeze()
        x = self.decoder(x)

        return x


def gen_padding_mask(lens_packed, padding_mask_idx):
    return padding_mask_idx[lens_packed]


def gen_precalcu_padding_mask_idx(total_len):
    return torch.triu(torch.ones(total_len, total_len) == 1)


def custom_padding_seq(sequences, max_len, padding_value=0.0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    # max_len = max([s.size(0) for s in sequences])

    out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[:length, i, ...] = tensor

    return out_tensor
