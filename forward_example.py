import itertools

import torch

from model import (
    TCA_SeqEncoder_TE,
    TCA_SeqEncoder_LSTM,
    TCA_FeatureEncoder,
    RAW_Feature_Encoder_1dcnn,
    RAW_Feature_Encoder_2dcnn,
    Seq_Classifier_TE,
    Seq_Classifier_LSTM,
)

use_tca = True
use_te_encoder = True
use_te_classifier = True
use_1dcnn = False
feature_dim = 64
subcarrier_num = 30
raw_feature_channel = 3
lstm_layer = 2
lstm_dim = 64
lstm2_layer = 2
lstm2_dim = 64
class_num = 6

music_feature_encoder = TCA_FeatureEncoder(in_feature_dim=4, out_feature_dim=64)
if use_te_encoder:
    music_lmax_encoder = TCA_SeqEncoder_TE(
        in_feature_dim=64, out_feature_dim=feature_dim, max_len=17
    )
else:
    music_lmax_encoder = TCA_SeqEncoder_LSTM(
        in_feature_dim=64,
        out_feature_dim=feature_dim,
        lstm_layer_num=lstm_layer,
        lstm_dim_num=lstm_dim,
    )


if use_1dcnn:
    raw_feature_encoder = RAW_Feature_Encoder_1dcnn(
        in_c=raw_feature_channel,
        out_feature_dim=feature_dim,
    )
else:
    raw_feature_encoder = RAW_Feature_Encoder_2dcnn(
        in_c=raw_feature_channel * 3 * subcarrier_num,
        out_feature_dim=feature_dim,
    )

if use_te_classifier:
    seq_classifier = Seq_Classifier_TE(
        in_feature_dim=feature_dim,
        class_nums=class_num,
        max_len=32,
    )
else:
    seq_classifier = Seq_Classifier_LSTM(
        in_feature_dim=feature_dim,
        lstm_layer_num=lstm2_layer,
        lstm_dim_num=lstm2_dim,
        class_nums=class_num,
    )


def forward(
    raw_batch=None,
    tca_batch=None,
):

    if use_tca:
        input_batch = tca_batch
        input_batch_concat = list(itertools.chain.from_iterable(input_batch))
        input_batch_concat_lmax_concat = torch.cat(input_batch_concat)
        input_batch_concat_lmax_feature = music_feature_encoder(
            input_batch_concat_lmax_concat
        )
        input_batch_concat_lmax_feature = torch.split(
            input_batch_concat_lmax_feature, [len(x) for x in input_batch_concat]
        )
        feature_out = music_lmax_encoder(input_batch_concat_lmax_feature)

    else:
        input_batch = raw_batch
        input_batch_concat = torch.cat(input_batch)
        feature_out = raw_feature_encoder(input_batch_concat)

    seq_in = torch.split(feature_out, [len(x) for x in input_batch])
    class_out = seq_classifier(seq_in)

    return class_out
