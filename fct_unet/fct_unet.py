import os
import torch
import torchvision.transforms.v2 as transforms
from fct import FC_Attention
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from utils import *


class UNetConfig:
    def __init__(
        self,
        input_resolution=(256, 256),
        num_channels=3,
        num_attention_heads=8,
        dropout=0.0,
        semantic_loss_ignore_index=255,
        query_projection_kernel_size=1,
        key_projection_kernel_size=3,
        value_projection_kernel_size=3,
        kv_kernel_size=3,
        head_unification_kernel_size=3,
        query_projection_stride=1,
        key_projection_stride=1,
        value_projection_stride=1,
        kv_stride=1,
        head_unification_stride=1,
        query_projection_padding=0,
        key_projection_padding=1,
        value_projection_padding=1,
        kv_padding=1,
        head_unification_padding=1,
        query_projection_dilation_factor=0,
        key_projection_dilation_factor=0,
        value_projection_dilation_factor=0,
        kv_dilation_factor=0,
        head_unification_dilation_factor=0,
        use_attention_bias=True,
        feed_forward_kernel_size=3,
        num_labels=21,
        max_channels=512,
    ):
        # For FC_Attention
        self.num_attention_heads = num_attention_heads
        self.query_projection_kernel_size = query_projection_kernel_size
        self.key_projection_kernel_size = key_projection_kernel_size
        self.value_projection_kernel_size = value_projection_kernel_size
        self.kv_kernel_size = kv_kernel_size
        self.head_unification_kernel_size = head_unification_kernel_size
        self.query_projection_stride = query_projection_stride
        self.key_projection_stride = key_projection_stride
        self.value_projection_stride = value_projection_stride
        self.kv_stride = kv_stride
        self.head_unification_stride = head_unification_stride
        self.query_projection_padding = query_projection_padding
        self.key_projection_padding = key_projection_padding
        self.value_projection_padding = value_projection_padding
        self.kv_padding = kv_padding
        self.head_unification_padding = head_unification_padding
        self.query_projection_dilation_factor = query_projection_dilation_factor
        self.key_projection_dilation_factor = key_projection_dilation_factor
        self.value_projection_dilation_factor = value_projection_dilation_factor
        self.kv_dilation_factor = kv_dilation_factor
        self.head_unification_dilation_factor = head_unification_dilation_factor
        self.use_attention_bias = use_attention_bias

        # For UNet
        self.feed_forward_kernel_size = feed_forward_kernel_size
        self.dropout = dropout

        # For Segmentor
        self.input_resolution = input_resolution
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.max_channels = max_channels

    def __call__(self, **kwargs):
        new_config = self.__dict__.copy()
        new_config.update(kwargs)
        return UNetConfig(**new_config)


# @torch.compile()
class UNetFeedForward(torch.nn.Module):
    def __init__(self, config, block_index=0, padding="same", input_channels=3, output_channels=64):
        super().__init__()
        self.config = config
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_index = block_index
        self.conv1 = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=self.config.feed_forward_kernel_size,
            padding=same_padding(self.config.feed_forward_kernel_size, format="single") if padding == "same" else 0,
        )
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(self.config.dropout)
        self.conv2 = torch.nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=self.config.feed_forward_kernel_size,
            padding=same_padding(self.config.feed_forward_kernel_size, format="single") if padding == "same" else 0,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


# @torch.compile()
class UNetTransformerEncoderBlock(torch.nn.Module):
    def __init__(self, config, block_index=0, input_channels=3, output_channels=64, input_resolution=None):
        super().__init__()
        self.config = config
        self.block_index = block_index
        self.input_resolution = input_resolution if input_resolution else config.input_resolution
        self.use_skip_connection = input_channels != output_channels
        self.layer_norm_1 = torch.nn.LayerNorm((input_channels, *self.input_resolution))
        self.attention = FC_Attention(
            embed_dim=input_channels,
            hidden_dim=input_channels,
            q_dim=input_channels,
            v_dim=input_channels,
            num_heads=1,
            internal_resolution=self.input_resolution,
            query_projection_kernel_size=self.config.query_projection_kernel_size,
            key_projection_kernel_size=self.config.key_projection_kernel_size,
            value_projection_kernel_size=self.config.value_projection_kernel_size,
            kv_kernel_size=self.config.kv_kernel_size,
            head_unification_kernel_size=self.config.head_unification_kernel_size,
            query_projection_stride=self.config.query_projection_stride,
            key_projection_stride=self.config.key_projection_stride,
            value_projection_stride=self.config.value_projection_stride,
            kv_stride=self.config.kv_stride,
            head_unification_stride=self.config.head_unification_stride,
            query_projection_padding=self.config.query_projection_padding,
            key_projection_padding=self.config.key_projection_padding,
            value_projection_padding=self.config.value_projection_padding,
            kv_padding=self.config.kv_padding,
            head_unification_padding=self.config.head_unification_padding,
            query_projection_dilation_factor=self.config.query_projection_dilation_factor,
            key_projection_dilation_factor=self.config.key_projection_dilation_factor,
            value_projection_dilation_factor=self.config.value_projection_dilation_factor,
            kv_dilation_factor=self.config.kv_dilation_factor,
            head_unification_dilation_factor=self.config.head_unification_dilation_factor,
            use_attention_bias=self.config.use_attention_bias,
        )
        self.layer_norm_2 = torch.nn.LayerNorm((input_channels, *self.input_resolution))
        self.feed_forward = UNetFeedForward(
            config,
            block_index=block_index,
            padding="same",
            input_channels=input_channels,
            output_channels=output_channels - input_channels if self.use_skip_connection else output_channels,
        )

    def forward(self, x):
        after_first_norm = self.layer_norm_1(x)
        if after_first_norm.isnan().any().item():
            print("NaN detected in layer norm 1")
        attn = self.attention(after_first_norm)
        x = self.layer_norm_2(x + attn)
        # NOTE: Using skip connections instead of residual connections to accomodate changing channel depths
        feed_forward_out = self.feed_forward(x)
        if self.use_skip_connection:
            x = torch.concat([x, feed_forward_out], dim=1)
        else:
            x = x + feed_forward_out
        x = torch.nn.functional.gelu(x)
        return x


# @torch.compile()
class UNetTransformerDecoderBlock(torch.nn.Module):
    def __init__(self, config, block_index=0, input_channels=3, output_channels=64, input_resolution=None):
        super().__init__()
        self.config = config
        self.input_resolution = input_resolution if input_resolution else config.input_resolution
        self.layer_norm_1 = torch.nn.LayerNorm((input_channels, *self.input_resolution))
        self.attention = FC_Attention(
            embed_dim=input_channels,
            hidden_dim=input_channels,
            q_dim=input_channels,
            v_dim=input_channels,
            num_heads=1,
            internal_resolution=self.input_resolution,
            query_projection_kernel_size=self.config.query_projection_kernel_size,
            key_projection_kernel_size=self.config.key_projection_kernel_size,
            value_projection_kernel_size=self.config.value_projection_kernel_size,
            kv_kernel_size=self.config.kv_kernel_size,
            head_unification_kernel_size=self.config.head_unification_kernel_size,
            query_projection_stride=self.config.query_projection_stride,
            key_projection_stride=self.config.key_projection_stride,
            value_projection_stride=self.config.value_projection_stride,
            kv_stride=self.config.kv_stride,
            head_unification_stride=self.config.head_unification_stride,
            query_projection_padding=self.config.query_projection_padding,
            key_projection_padding=self.config.key_projection_padding,
            value_projection_padding=self.config.value_projection_padding,
            kv_padding=self.config.kv_padding,
            head_unification_padding=self.config.head_unification_padding,
            query_projection_dilation_factor=self.config.query_projection_dilation_factor,
            key_projection_dilation_factor=self.config.key_projection_dilation_factor,
            value_projection_dilation_factor=self.config.value_projection_dilation_factor,
            kv_dilation_factor=self.config.kv_dilation_factor,
            head_unification_dilation_factor=self.config.head_unification_dilation_factor,
            use_attention_bias=self.config.use_attention_bias,
        )
        self.layer_norm_2 = torch.nn.LayerNorm((input_channels, *self.input_resolution))
        self.feed_forward = UNetFeedForward(
            config,
            block_index=block_index,
            padding="same",
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x):
        input_x = x
        after_first_norm = self.layer_norm_1(x)
        if after_first_norm.isnan().any().item():
            print("NaN detected in layer norm 1")
        attn = self.attention(after_first_norm)
        if attn.isnan().any().item():
            print("NaN detected in attn")
        x = self.layer_norm_2(x + attn)
        if x.isnan().any().item():
            print("NaN detected in x")
        # NOTE: Using channel max pooling to allow residual connections with channel reduction
        x_reduced = torch.nn.functional.adaptive_max_pool3d(x, (x.shape[1] // 2, x.shape[2], x.shape[3]))
        if x_reduced.isnan().any().item():
            print("NaN detected in x_reduced")
        x = x_reduced + self.feed_forward(x)
        if x.isnan().any().item():
            print("NaN detected in x")
        x = torch.nn.functional.gelu(x)
        if x.isnan().any().item():
            print("NaN detected in x")
        return x


# @torch.compile()
class UNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_resolution = config.input_resolution
        self.max_channels = config.max_channels
        self.resolutions = [tuple(map(lambda x: x // (2**i), self.input_resolution)) for i in range(5)]

        self.depths = [self.max_channels // (2**i) for i in range(4, -1, -1)]
        self.embedding = torch.nn.Conv2d(3, self.depths[0], kernel_size=7, padding=same_padding(7, format="single"))
        self.learned_positional_bias = torch.nn.Parameter(torch.zeros((1, self.depths[0], *self.resolutions[0])))
        self.periodic_positional_encoding = PositionalEncodingPermute2D(self.depths[0])

        self.encoder_0 = UNetTransformerEncoderBlock(
            config,
            input_resolution=self.resolutions[0],
            input_channels=self.depths[0],
            output_channels=self.depths[0],
            block_index="encoder_0",
        )
        self.encoder_1 = UNetTransformerEncoderBlock(
            config,
            input_resolution=self.resolutions[1],
            input_channels=self.depths[0],
            output_channels=self.depths[1],
            block_index="encoder_1",
        )
        self.encoder_2 = UNetTransformerEncoderBlock(
            config,
            input_resolution=self.resolutions[2],
            input_channels=self.depths[1],
            output_channels=self.depths[2],
            block_index="encoder_2",
        )
        self.encoder_3 = UNetTransformerEncoderBlock(
            config,
            input_resolution=self.resolutions[3],
            input_channels=self.depths[2],
            output_channels=self.depths[3],
            block_index="encoder_3",
        )
        self.encoder_4 = UNetTransformerEncoderBlock(
            config,
            input_resolution=self.resolutions[4],
            input_channels=self.depths[3],
            output_channels=self.depths[4],
            block_index="encoder_4",
        )

        self.upsampler_0 = torch.nn.ConvTranspose2d(self.depths[4], self.depths[3], kernel_size=2, stride=2)
        self.decoder_0 = UNetTransformerDecoderBlock(
            config,
            input_resolution=self.resolutions[3],
            input_channels=self.depths[4],
            output_channels=self.depths[3],
            block_index="decoder_0",
        )
        self.upsampler_1 = torch.nn.ConvTranspose2d(self.depths[3], self.depths[2], kernel_size=2, stride=2)
        self.decoder_1 = UNetTransformerDecoderBlock(
            config,
            input_resolution=self.resolutions[2],
            input_channels=self.depths[3],
            output_channels=self.depths[2],
            block_index="decoder_1",
        )
        self.upsampler_2 = torch.nn.ConvTranspose2d(self.depths[2], self.depths[1], kernel_size=2, stride=2)
        self.decoder_2 = UNetTransformerDecoderBlock(
            config,
            input_resolution=self.resolutions[1],
            input_channels=self.depths[2],
            output_channels=self.depths[1],
            block_index="decoder_2",
        )
        self.upsampler_3 = torch.nn.ConvTranspose2d(self.depths[1], self.depths[0], kernel_size=2, stride=2)
        self.decoder_3 = UNetTransformerDecoderBlock(
            config,
            input_resolution=self.resolutions[0],
            input_channels=self.depths[1],
            output_channels=self.depths[0],
            block_index="decoder_3",
        )

    def forward(self, x):
        B = x.shape[0]

        embeddings = self.embedding(x)

        # Add positional embedding (periodic augmented by learned positional bias)
        periodic_positional_encoding = self.periodic_positional_encoding(embeddings)
        learned_positional_bias = self.learned_positional_bias.repeat(B, 1, 1, 1)
        embeddings = embeddings + periodic_positional_encoding + learned_positional_bias

        encoder_0_out = self.encoder_0(embeddings)
        # assert_shape(encoder_0_out, (B, self.depths[0], *self.resolutions[0]))
        encoder_0_pooled = torch.nn.functional.max_pool2d(encoder_0_out, kernel_size=2, stride=2)
        # assert_shape(encoder_0_pooled, (B, self.depths[0], *self.resolutions[1]))

        encoder_1_out = self.encoder_1(encoder_0_pooled)
        # assert_shape(encoder_1_out, (B, self.depths[1], *self.resolutions[1]))
        encoder_1_pooled = torch.nn.functional.max_pool2d(encoder_1_out, kernel_size=2, stride=2)
        # assert_shape(encoder_1_pooled, (B, self.depths[1], *self.resolutions[2]))

        encoder_2_out = self.encoder_2(encoder_1_pooled)
        # assert_shape(encoder_2_out, (B, self.depths[2], *self.resolutions[2]))
        encoder_2_pooled = torch.nn.functional.max_pool2d(encoder_2_out, kernel_size=2, stride=2)
        # assert_shape(encoder_2_pooled, (B, self.depths[2], *self.resolutions[3]))

        encoder_3_out = self.encoder_3(encoder_2_pooled)
        # assert_shape(encoder_3_out, (B, self.depths[3], *self.resolutions[3]))
        encoder_3_pooled = torch.nn.functional.max_pool2d(encoder_3_out, kernel_size=2, stride=2)
        # assert_shape(encoder_3_pooled, (B, self.depths[3], *self.resolutions[4]))

        encoder_4_out = self.encoder_4(encoder_3_pooled)
        # assert_shape(encoder_4_out, (B, self.depths[4], *self.resolutions[4]))
        encoder_4_upsampled = self.upsampler_0(encoder_4_out)
        # assert_shape(encoder_4_upsampled, (B, self.depths[3], *self.resolutions[3]))

        decoder_0_in = torch.concat([encoder_4_upsampled, encoder_3_out], dim=1)
        decoder_0_out = self.decoder_0(decoder_0_in)
        # assert_shape(decoder_0_out, (B, self.depths[3], *self.resolutions[3]))
        decoder_0_upsampled = self.upsampler_1(decoder_0_out)
        # assert_shape(decoder_0_upsampled, (B, self.depths[2], *self.resolutions[2]))

        decoder_1_in = torch.concat([decoder_0_upsampled, encoder_2_out], dim=1)
        decoder_1_out = self.decoder_1(decoder_1_in)
        # assert_shape(decoder_1_out, (B, self.depths[2], *self.resolutions[2]))
        decoder_1_upsampled = self.upsampler_2(decoder_1_out)
        # assert_shape(decoder_1_upsampled, (B, self.depths[1], *self.resolutions[1]))

        decoder_2_in = torch.concat([decoder_1_upsampled, encoder_1_out], dim=1)
        decoder_2_out = self.decoder_2(decoder_2_in)
        # assert_shape(decoder_2_out, (B, self.depths[1], *self.resolutions[1]))
        decoder_2_upsampled = self.upsampler_3(decoder_2_out)
        # assert_shape(decoder_2_upsampled, (B, self.depths[0], *self.resolutions[0]))

        decoder_3_in = torch.concat([decoder_2_upsampled, encoder_0_out], dim=1)
        decoder_3_out = self.decoder_3(decoder_3_in)
        # assert_shape(decoder_3_out, (B, self.depths[0], *self.resolutions[0]))

        return decoder_3_out


class UNetForSemanticSegmentation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.unet = UNet(config)
        self.segmentation_head = torch.nn.Conv2d(self.unet.depths[0], config.num_labels, kernel_size=1)

    def forward(self, x):
        x = self.unet(x)
        x = self.segmentation_head(x)
        return x


def test():
    import time

    instantiation_start = time.time()
    unet = UNet(UNetConfig()).to("cuda")
    dummy_input = torch.randn(4, 3, 256, 256).to("cuda")
    print(f"Instantiation time: {time.time() - instantiation_start:.2f}s")
    print(f"Number of parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    processing_start = time.time()
    for _ in range(10):
        dummy_output = unet(dummy_input)
        print("Output shape:", dummy_output.shape)
    print(f"Processing time: {time.time() - processing_start:.2f}s")


if __name__ == "__main__":
    test()
