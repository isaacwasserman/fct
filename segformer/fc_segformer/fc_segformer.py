import os
from transformers.models.segformer.modeling_segformer import (
    SegformerConfig,
    SegformerDropPath,
    SegformerEfficientSelfAttention,
    SegformerLayer,
    SegformerModel,
    SegformerOverlapPatchEmbeddings,
    SegformerSelfOutput,
    SegformerAttention,
    SegformerEncoder,
)
from transformers import SegformerForSemanticSegmentation
from transformers.activations import ACT2FN

import torch.nn as nn
from fct import FC_Attention

from utils import *
import torch


class FC_SegformerEfficientSelfAttention(SegformerEfficientSelfAttention):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio, height, width):
        super().__init__(config, hidden_size, num_attention_heads, sequence_reduction_ratio)
        self.fc_attention = FC_Attention(
            embed_dim=hidden_size,
            hidden_dim=hidden_size,
            q_dim=hidden_size,
            v_dim=hidden_size,
            num_heads=num_attention_heads,
            internal_resolution=(height, width),
            block_index=0,
            kernel_size=config.attention_kernel_size,
        )

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
    ):
        hidden_states = hidden_states.view(-1, height, width, self.hidden_size).permute(0, 3, 1, 2)
        context_layer = self.fc_attention(hidden_states)
        context_layer = context_layer.permute(0, 2, 3, 1).contiguous()
        context_layer = context_layer.reshape(-1, height * width, self.hidden_size)
        outputs = (context_layer,)
        return outputs


class FC_SegformerAttention(SegformerAttention):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio, height, width):
        super().__init__(config, hidden_size, num_attention_heads, sequence_reduction_ratio)
        self.self = FC_SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            height=height,
            width=width,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class FC_SegformerMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.conv1 = nn.Conv2d(
            in_features,
            hidden_features,
            kernel_size=config.feedforward_kernel_size,
            padding=same_padding(config.feedforward_kernel_size, "single"),
        )
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=config.feedforward_kernel_size,
            padding=same_padding(config.feedforward_kernel_size, "single"),
            groups=hidden_features,
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.conv2 = nn.Conv2d(
            hidden_features,
            out_features,
            kernel_size=config.feedforward_kernel_size,
            padding=same_padding(config.feedforward_kernel_size, "single"),
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = hidden_states.reshape(-1, height, width, hidden_states.shape[-1]).permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = hidden_states.view(-1, height * width, hidden_states.shape[-1])
        return hidden_states


class FC_SegformerLayer(SegformerLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio, height, width):
        super().__init__(config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        self.attention = FC_SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            height=height,
            width=width,
        )
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = FC_SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class FC_SegformerEncoder(SegformerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        heights = []
        widths = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
            if i == 0:
                H, W = self.config.input_resolution
            else:
                H, W = heights[i - 1], widths[i - 1]
            K = config.patch_sizes[i]
            P = K // 2
            S = config.strides[i]
            height = ((H - K + (2 * P)) // S) + 1
            width = ((W - K + (2 * P)) // S) + 1
            heights.append(height)
            widths.append(width)
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    FC_SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                        height=heights[i],
                        width=widths[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList([nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)])


class FC_SegformerModel(SegformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = FC_SegformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()


class FC_SegformerMLP(nn.Module):
    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            input_dim,
            config.decoder_hidden_size,
            kernel_size=config.feedforward_kernel_size,
            padding=same_padding(config.feedforward_kernel_size, "single"),
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = hidden_states.view(hidden_states.size(0), -1, hidden_states.size(-1))
        return hidden_states


class FC_SegformerDecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = FC_SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=config.decoder_kernel_size,
            padding=same_padding(config.decoder_kernel_size, "single"),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(
            config.decoder_hidden_size,
            config.num_labels,
            kernel_size=config.decoder_kernel_size,
            padding=same_padding(config.decoder_kernel_size, "single"),
        )

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


class FC_SegformerForSemanticSegmentation(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = FC_SegformerModel(config)
        self.decode_head = FC_SegformerDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()


class FC_SegformerConfig(SegformerConfig):
    def __init__(
        self,
        input_resolution=(256, 256),
        attention_kernel_size=1,
        feedforward_kernel_size=3,
        decoder_kernel_size=5,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        semantic_loss_ignore_index=255,
        **kwargs,
    ):
        super().__init__(
            num_channels,
            num_encoder_blocks,
            depths,
            sr_ratios,
            hidden_sizes,
            patch_sizes,
            strides,
            num_attention_heads,
            mlp_ratios,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            classifier_dropout_prob,
            initializer_range,
            drop_path_rate,
            layer_norm_eps,
            decoder_hidden_size,
            semantic_loss_ignore_index,
            **kwargs,
        )
        self.input_resolution = input_resolution
        self.attention_kernel_size = attention_kernel_size
        self.feedforward_kernel_size = feedforward_kernel_size
        self.decoder_kernel_size = decoder_kernel_size
