import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from compressai.models.google import MeanScaleHyperprior

from compressai.layers import (
    GDN,
    MaskedConv2d,
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)


class CrossScale_PLYR(MeanScaleHyperprior):
    """
    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        # self.g_a = nn.Sequential(
        #     ResidualBlockWithStride(256, N, stride=2),
        #     ResidualBlock(N, N),
        #     ResidualBlockWithStride(N, N, stride=2),
        #     AttentionBlock(N),
        #     ResidualBlock(N, N),
        #     ResidualBlockWithStride(N, N, stride=2),
        #     ResidualBlock(N, N),
        #     conv3x3(N, N, stride=2),
        #     AttentionBlock(N),
        # )

        channels_plyr = [256, 256, 256, 256]

        self.g_a_plyr = nn.ModuleDict(
            {
                "p2": nn.Sequential(
                    ResidualBlockWithStride(channels_plyr[0], N, stride=2),
                    ResidualBlock(N, N),
                    ResidualBlockWithStride(N, N, stride=2),
                    ResidualBlock(N, N // 2),
                    conv3x3(N // 2, N // 4, stride=2),
                    AttentionBlock(N // 4),
                ),
                "p3": nn.Sequential(
                    ResidualBlockWithStride(channels_plyr[1], N, stride=2),
                    ResidualBlock(N, N // 2),
                    conv3x3(N // 2, N // 4, stride=2),
                    AttentionBlock(N // 4),
                ),
                "p4": nn.Sequential(
                    ResidualBlockWithStride(channels_plyr[2], N, stride=2),
                    ResidualBlock(N, N // 2),
                    conv3x3(N // 2, N // 4),
                    AttentionBlock(N // 4),
                ),
                "p5": nn.Sequential(
                    ResidualBlock(channels_plyr[3], N // 2),
                    conv3x3(N // 2, N // 4),
                    AttentionBlock(N // 4),
                ),
            }
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        # self.g_s = nn.Sequential(
        #     AttentionBlock(N),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     AttentionBlock(N),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     subpel_conv3x3(N, 256, 2),
        # )


        self.g_s_plyr = nn.ModuleDict(
            {
                "p2": nn.Sequential(
                    AttentionBlock(N),
                    ResidualBlockUpsample(N, N, 2),
                    ResidualBlock(N, N),
                    ResidualBlockUpsample(N, N, 2),
                    ResidualBlock(N, N),
                    subpel_conv3x3(N, 256, 2),
                    
                ),
                "p3": nn.Sequential(
                    AttentionBlock(N),
                    ResidualBlockUpsample(N, N, 2),
                    ResidualBlock(N, N),
                    subpel_conv3x3(N, 256, 2),
                ),
                "p4": nn.Sequential(
                    AttentionBlock(N),
                    ResidualBlockUpsample(N, N, 2),
                    ResidualBlock(N, 256),
                ),
                "p5": nn.Sequential(
                    AttentionBlock(N),
                    ResidualBlock(N, 256),
                ),
            }
        )


        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        M = state_dict["context_prediction.0.conv1.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):

        y_plyr = {}
        for layer_name, feature in x.items():
            y_plyr[layer_name] = self.g_a_plyr[layer_name](feature)
        y = torch.cat(list(y_plyr.values()), axis=1)

        # y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)


        x_hat={}
        for layer_name, g_a_s in self.g_s_plyr.items():
            x_hat[layer_name] = g_a_s(y_hat)

        #x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
