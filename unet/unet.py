import os
import torch
import torchvision.transforms.v2 as transforms
from fct import FC_Attention

# from pascal_utils import PascalTrainer
from utils import *


class UNetConfig:
    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        attention_kernel_size=1,
        dropout=0.0,
        feed_forward_kernel_size=3,
        num_labels=21,
        input_resolution=(256, 256),
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_kernel_size = attention_kernel_size
        self.dropout = dropout
        self.feed_forward_kernel_size = feed_forward_kernel_size
        self.num_labels = num_labels
        self.input_resolution = input_resolution

    def __call__(self, **kwargs):
        new_config = self.__dict__.copy()
        new_config.update(kwargs)
        return UNetConfig(**new_config)


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


class UNetTransformerEncoderBlock(torch.nn.Module):
    def __init__(self, config, block_index=0, input_channels=3, output_channels=64, input_resolution=None):
        super().__init__()
        self.config = config
        self.input_resolution = input_resolution if input_resolution else config.input_resolution
        self.layer_norm_1 = torch.nn.LayerNorm((input_channels, *self.input_resolution))
        self.attention = FC_Attention(
            embed_dim=input_channels,
            q_dim=input_channels,
            v_dim=input_channels,
            num_heads=1,
            internal_resolution=self.input_resolution,
            block_index=block_index,
            kernel_size=self.config.attention_kernel_size,
        )
        self.layer_norm_2 = torch.nn.LayerNorm((input_channels, *self.input_resolution))
        self.feed_forward = UNetFeedForward(
            config,
            block_index=block_index,
            padding="same",
            input_channels=input_channels,
            output_channels=output_channels - input_channels,
        )

    def forward(self, x):
        attn = self.attention(self.layer_norm_1(x))
        x = self.layer_norm_2(x + attn)
        # NOTE: Using skip connections instead of residual connections to accomodate changing channel depths
        x = torch.concat([x, self.feed_forward(x)], dim=1)
        return x


class UNetTransformerDecoderBlock(torch.nn.Module):
    def __init__(self, config, block_index=0, input_channels=3, output_channels=64, input_resolution=None):
        super().__init__()
        self.config = config
        self.input_resolution = input_resolution if input_resolution else config.input_resolution
        self.layer_norm_1 = torch.nn.LayerNorm((input_channels, *self.input_resolution))
        self.attention = FC_Attention(
            embed_dim=input_channels,
            q_dim=input_channels,
            v_dim=input_channels,
            num_heads=1,
            internal_resolution=self.input_resolution,
            block_index=block_index,
            kernel_size=self.config.attention_kernel_size,
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
        attn = self.attention(self.layer_norm_1(x))
        x = self.layer_norm_2(x + attn)
        # NOTE: Using channel max pooling to allow residual connections with channel reduction
        x_reduced = torch.nn.functional.adaptive_max_pool3d(x, (x.shape[1] // 2, x.shape[2], x.shape[3]))
        x = x_reduced + self.feed_forward(x)
        return x


class UNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_resolution = config.input_resolution
        self.resolutions = [tuple(map(lambda x: x // (2**i), self.input_resolution)) for i in range(5)]
        self.encoder_0 = UNetTransformerEncoderBlock(
            config, input_resolution=self.resolutions[0], input_channels=3, output_channels=64
        )
        self.encoder_1 = UNetTransformerEncoderBlock(
            config, input_resolution=self.resolutions[1], input_channels=64, output_channels=128
        )
        self.encoder_2 = UNetTransformerEncoderBlock(
            config, input_resolution=self.resolutions[2], input_channels=128, output_channels=256
        )
        self.encoder_3 = UNetTransformerEncoderBlock(
            config, input_resolution=self.resolutions[3], input_channels=256, output_channels=512
        )
        self.encoder_4 = UNetTransformerEncoderBlock(
            config, input_resolution=self.resolutions[4], input_channels=512, output_channels=1024
        )

        self.upsampler_0 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_0 = UNetTransformerDecoderBlock(
            config, input_resolution=self.resolutions[3], input_channels=1024, output_channels=512, block_index="decoder_0"
        )
        self.upsampler_1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_1 = UNetTransformerDecoderBlock(
            config, input_resolution=self.resolutions[2], input_channels=512, output_channels=256, block_index="decoder_1"
        )
        self.upsampler_2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_2 = UNetTransformerDecoderBlock(
            config, input_resolution=self.resolutions[1], input_channels=256, output_channels=128, block_index="decoder_2"
        )
        self.upsampler_3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_3 = UNetTransformerDecoderBlock(
            config, input_resolution=self.resolutions[0], input_channels=128, output_channels=64, block_index="decoder_3"
        )

        self.segmentation_head = torch.nn.Conv2d(64, config.num_labels, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        encoder_0_out = self.encoder_0(x)
        assert_shape(encoder_0_out, (B, 64, *self.resolutions[0]))
        encoder_0_pooled = torch.nn.functional.max_pool2d(encoder_0_out, kernel_size=2, stride=2)
        assert_shape(encoder_0_pooled, (B, 64, *self.resolutions[1]))

        encoder_1_out = self.encoder_1(encoder_0_pooled)
        assert_shape(encoder_1_out, (B, 128, *self.resolutions[1]))
        encoder_1_pooled = torch.nn.functional.max_pool2d(encoder_1_out, kernel_size=2, stride=2)
        assert_shape(encoder_1_pooled, (B, 128, *self.resolutions[2]))

        encoder_2_out = self.encoder_2(encoder_1_pooled)
        assert_shape(encoder_2_out, (B, 256, *self.resolutions[2]))
        encoder_2_pooled = torch.nn.functional.max_pool2d(encoder_2_out, kernel_size=2, stride=2)
        assert_shape(encoder_2_pooled, (B, 256, *self.resolutions[3]))

        encoder_3_out = self.encoder_3(encoder_2_pooled)
        assert_shape(encoder_3_out, (B, 512, *self.resolutions[3]))
        encoder_3_pooled = torch.nn.functional.max_pool2d(encoder_3_out, kernel_size=2, stride=2)
        assert_shape(encoder_3_pooled, (B, 512, *self.resolutions[4]))

        encoder_4_out = self.encoder_4(encoder_3_pooled)
        assert_shape(encoder_4_out, (B, 1024, *self.resolutions[4]))
        encoder_4_upsampled = self.upsampler_0(encoder_4_out)
        assert_shape(encoder_4_upsampled, (B, 512, *self.resolutions[3]))

        decoder_0_in = torch.concat([encoder_4_upsampled, encoder_3_out], dim=1)
        decoder_0_out = self.decoder_0(decoder_0_in)
        assert_shape(decoder_0_out, (B, 512, *self.resolutions[3]))
        decoder_0_upsampled = self.upsampler_1(decoder_0_out)
        assert_shape(decoder_0_upsampled, (B, 256, *self.resolutions[2]))

        decoder_1_in = torch.concat([decoder_0_upsampled, encoder_2_out], dim=1)
        decoder_1_out = self.decoder_1(decoder_1_in)
        assert_shape(decoder_1_out, (B, 256, *self.resolutions[2]))
        decoder_1_upsampled = self.upsampler_2(decoder_1_out)
        assert_shape(decoder_1_upsampled, (B, 128, *self.resolutions[1]))

        decoder_2_in = torch.concat([decoder_1_upsampled, encoder_1_out], dim=1)
        decoder_2_out = self.decoder_2(decoder_2_in)
        assert_shape(decoder_2_out, (B, 128, *self.resolutions[1]))
        decoder_2_upsampled = self.upsampler_3(decoder_2_out)
        assert_shape(decoder_2_upsampled, (B, 64, *self.resolutions[0]))

        decoder_3_in = torch.concat([decoder_2_upsampled, encoder_0_out], dim=1)
        decoder_3_out = self.decoder_3(decoder_3_in)
        assert_shape(decoder_3_out, (B, 64, *self.resolutions[0]))

        segmentation = self.segmentation_head(decoder_3_out)
        assert_shape(segmentation, (B, self.config.num_labels, *self.resolutions[0]))

        return segmentation


if __name__ == "__main__":
    unet = UNet(UNetConfig()).to("cuda")
    # print number of parameters
    for name, param in unet.named_parameters():
        print(f"Layer: {name} | Size: {param.size()}")
    print(f"Number of parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
    dummy_input = torch.randn(1, 3, 256, 256).to("cuda")
    dummy_output = unet(dummy_input)
    assert_shape(dummy_output, (1, 21, 256, 256))
    print("UNet test passed!")
