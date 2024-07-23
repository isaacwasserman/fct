import os
from segformer_arch import *
from fct import FC_Attention

from pascal_utils import PascalTrainer, get_dataset
import torch


class FC_SegformerEfficientSelfAttention(SegformerEfficientSelfAttention):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__(config, hidden_size, num_attention_heads, sequence_reduction_ratio)
        self.fc_attention = FC_Attention(
            embed_dim=hidden_size,
            hidden_dim=hidden_size,
            q_dim=hidden_size,
            v_dim=hidden_size,
            num_heads=num_attention_heads,
            internal_resolution=(256, 256),
            block_index=0,
            kernel_size=1,
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
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__(config, hidden_size, num_attention_heads, sequence_reduction_ratio)
        self.self = FC_SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()


class FC_SegformerMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.conv2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
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

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__(config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        self.attention = FC_SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
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
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
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


class FC_SegformerForSemanticSegmentation(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = FC_SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()


device = "cpu"

config = SegformerConfig(num_labels=21)
model = FC_SegformerForSemanticSegmentation(config)
model = model.to(device)

dummy_image = torch.randn(4, 3, 256, 256).to(device)
output = model(dummy_image)

print(output.logits.shape)
