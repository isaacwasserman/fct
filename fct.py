import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from utils import *


class FC_Attention(torch.nn.Module):
    def __init__(
        self,
        embed_dim=256,
        hidden_dim=512,
        q_dim=512,
        v_dim=256,
        num_heads=8,
        dropout=0.0,
        block_index=0,
        internal_resolution=(32, 32),
        kernel_size=1,
    ):
        super().__init__()
        self.q_net = torch.nn.Conv2d(
            embed_dim, q_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.k_net = torch.nn.Conv2d(
            embed_dim, q_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.v_net = torch.nn.Conv2d(
            embed_dim, v_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.bias_net = torch.nn.Sequential(
            torch.nn.Conv2d(embed_dim, v_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")),
            torch.nn.AvgPool2d(kernel_size=internal_resolution),
        )
        self.bias_norm = torch.nn.LayerNorm(v_dim)
        self.head_unification = torch.nn.Conv2d(
            v_dim, embed_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.block_index = block_index
        self.internal_resolution = internal_resolution

    def break_into_heads(self, M):
        B, D, H, W = M.shape
        h = self.num_heads
        return M.reshape(B, h, D // h, H, W).flatten(0, 1)

    def sum_pool_to_resolution(self, x, output_resolution=3):
        # Temporarily changing interpolation mode to bilinear because of torch bug
        # a = torch.nn.functional.interpolate(x, size=(output_resolution, output_resolution), mode="nearest")
        a = torch.nn.functional.interpolate(x, size=(output_resolution, output_resolution), mode="bilinear")
        # a = torch.nn.functional.adaptive_avg_pool2d(x, output_resolution)
        a = a * (output_resolution**2)
        return a

    def spatial_linear_self_attention(self, x, Q, K, V):
        """
        Q: (B, Dq, H, W)
        K: (B, Dq, H, W)
        V: (B, Dv, H, W)
        """

        B, Dq, H, W = Q.shape
        _, Dv, _, _ = V.shape
        _, Dm, _, _ = x.shape
        h = self.num_heads

        assert_shape(x, (B, Dm, H, W))
        assert_shape(Q, (B, Dq, H, W))
        assert_shape(K, (B, Dq, H, W))
        assert_shape(V, (B, Dv, H, W))

        # Compute softmax of Q over channel dimension
        Q = Q - Q.max(dim=-3, keepdim=True)[0]
        Q = torch.exp(torch.nn.functional.log_softmax(Q, dim=-3))
        # Q = torch.nn.functional.softmax(Q, dim=-3)
        assert_shape(Q, (B, Dq, H, W))

        # Compute softmax of K over both spatial dimensions simultaneously
        K = K.flatten(-2)
        K = K - K.max(dim=-1, keepdim=True)[0]
        K = torch.exp(torch.nn.functional.log_softmax(K.flatten(-2), dim=-1)).reshape(B, Dq, H, W)
        # K = torch.nn.functional.softmax(K.flatten(-2), dim=-1).reshape(B, Dq, H, W)
        assert_shape(K, (B, Dq, H, W))

        # Break Q, K, V into heads
        Q = self.break_into_heads(Q)
        assert_shape(Q, (B * h, Dq // h, H, W))
        K = self.break_into_heads(K)
        assert_shape(K, (B * h, Dq // h, H, W))
        V = self.break_into_heads(V)
        assert_shape(V, (B * h, Dv // h, H, W))

        # Old 1x1 KV calculation
        # ---------------------------------------
        # # Compute K^T @ V (kind of)
        # # i.e. outer product of K and V, summed over spatial dimensions
        # (Bh, Dq//h, 1, H, W) x (Bh, 1, Dv//h, H, W) -> (Bh, Dq//h, Dv//h, H, W) -> (Bh, Dq//h, Dv//h)
        # KV = (K.unsqueeze(2) * V.unsqueeze(1)).sum(dim=[-1, -2])
        # assert_shape(KV, (B * h, Dq // h, Dv // h))

        # # Reshape KV into 2D convolutional kernels
        # KV = KV.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 1, 3, 4).flatten(0, 1)
        # assert_shape(KV, (B * h * (Dv // h), Dq // h, 1, 1))
        # ---------------------------------------

        # New kxk KV calculation
        # ---------------------------------------

        KV = K.unsqueeze(2) * V.unsqueeze(1)
        if KV.flatten().isinf().any().item():
            print(f"KV inf: {torch.isinf(KV.flatten()).any()}")
        if KV.flatten().isnan().any().item():
            print(f"KV nan: {torch.isnan(KV.flatten()).any()}")
        assert_shape(KV, (B * h, Dq // h, Dv // h, H, W))
        KV = KV.flatten(0, 1)
        assert_shape(KV, (B * h * Dq // h, Dv // h, H, W))
        KV = self.sum_pool_to_resolution(KV, output_resolution=self.kernel_size)
        assert_shape(KV, (B * h * Dq // h, Dv // h, self.kernel_size, self.kernel_size))
        KV = KV.reshape(B * h, Dq // h, Dv // h, self.kernel_size, self.kernel_size)
        assert_shape(KV, (B * h, Dq // h, Dv // h, self.kernel_size, self.kernel_size))
        KV = KV.permute(0, 2, 1, 3, 4)
        assert_shape(KV, (B * h, Dv // h, Dq // h, self.kernel_size, self.kernel_size))
        KV = KV.flatten(0, 1)
        assert_shape(KV, (B * h * (Dv // h), Dq // h, self.kernel_size, self.kernel_size))

        # Reshape Q into a single B * Dq channel image
        Q = Q.flatten(0, 1).unsqueeze(0)
        assert_shape(Q, (1, B * h * (Dq // h), H, W))
        bias = self.bias_net(x)
        bias = self.bias_norm(bias.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        assert_shape(bias, (B, Dv, 1, 1))
        if bias.isinf().any().item():
            print(f"bias inf: {torch.isinf(bias).any()}")
        bias = bias.flatten()
        assert_shape(bias, (B * h * (Dv // h),))
        if bias.isinf().any().item():
            print(f"bias inf: {torch.isinf(bias).any()}")

        # QKV is grouped (B*h groups) convolution of Q with KV
        QKV = torch.nn.functional.conv2d(Q, KV, bias=bias, groups=B * h, padding=same_padding(KV.shape[-1], format="single"))
        assert_shape(QKV, (1, B * h * Dv // h, H, W))
        QKV = QKV.view(B, Dv, H, W)
        assert_shape(QKV, (B, Dv, H, W))

        # Unify heads
        QKV = self.head_unification(QKV)

        return QKV

    def forward(self, x):
        B, D, H, W = x.shape
        Q = self.q_net(x)
        K = self.k_net(x)
        V = self.v_net(x)
        attn = self.spatial_linear_self_attention(x, Q, K, V)
        assert_shape(attn, (B, D, H, W))
        return attn


# Define transformer module
class FC_TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim=256,
        hidden_dim=512,
        q_dim=512,
        v_dim=256,
        num_heads=8,
        dropout=0.0,
        internal_resolution=(32, 32),
        block_index=0,
        kernel_size=1,
    ):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = torch.nn.LayerNorm((embed_dim, *internal_resolution))
        # self.q_net = torch.nn.Conv2d(
        #     embed_dim, q_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        # )
        # self.k_net = torch.nn.Conv2d(
        #     embed_dim, q_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        # )
        # self.v_net = torch.nn.Conv2d(
        #     embed_dim, v_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        # )
        # self.bias_net = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=internal_resolution, padding=0, groups=embed_dim)
        # self.head_unification = torch.nn.Conv2d(
        #     v_dim, embed_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        # )
        self.attention = FC_Attention(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            q_dim=q_dim,
            v_dim=v_dim,
            num_heads=num_heads,
            dropout=dropout,
            internal_resolution=internal_resolution,
            block_index=block_index,
            kernel_size=kernel_size,
        )
        self.layer_norm_2 = torch.nn.LayerNorm((embed_dim, *internal_resolution))
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Conv2d(embed_dim, hidden_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(hidden_dim, embed_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")),
            torch.nn.Dropout(dropout),
        )
        self.num_heads = num_heads
        self.block_index = block_index
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

    def forward(self, x):
        B, D, H, W = x.shape
        Dq = self.q_dim
        Dv = self.v_dim
        after_norm_1 = self.layer_norm_1(x)

        attn = self.attention(after_norm_1)
        assert_shape(attn, (B, D, H, W))

        x = x + attn
        assert_shape(x, (B, D, H, W))

        x = self.layer_norm_2(x)
        assert_shape(x, (B, D, H, W))

        x = x + self.feed_forward(x)
        assert_shape(x, (B, D, H, W))
        return x


# Define Vision Transformer
class FullyConvolutionalTransformer(torch.nn.Module):
    def __init__(
        self,
        embed_dim=256,
        hidden_dim=512,
        q_dim=512,
        v_dim=256,
        num_channels=3,
        num_heads=8,
        num_layers=6,
        num_classes=10,
        dropout=0.0,
        patch_equivalent_mode=True,
        patch_width=4,
        input_resolution=(32, 32),
        transformer_kernel_size=1,
        **kwargs,
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        if patch_equivalent_mode:
            embedding_kernel_size = patch_width
            stride = patch_width
            internal_resolution = (input_resolution[0] // patch_width, input_resolution[1] // patch_width)
        else:
            embedding_kernel_size = 1
            stride = 1
            internal_resolution = input_resolution
        if not isinstance(transformer_kernel_size, list):
            transformer_kernel_size = [transformer_kernel_size] * num_layers
        elif len(transformer_kernel_size) < num_layers:
            transformer_kernel_size = transformer_kernel_size + [transformer_kernel_size[-1]] * (
                num_layers - len(transformer_kernel_size)
            )
        self.input_layer_cnn = torch.nn.Sequential(
            torch.nn.ZeroPad2d(same_padding(embedding_kernel_size) if not patch_equivalent_mode else 0),
            torch.nn.Conv2d(
                num_channels, num_channels, kernel_size=embedding_kernel_size, stride=stride, padding=0, groups=num_channels
            ),
            torch.nn.Conv2d(num_channels, embed_dim, kernel_size=1, stride=1, padding=0),
            torch.nn.GELU(),
        )
        self.transformer = torch.nn.Sequential(
            *(
                FC_TransformerBlock(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    q_dim=q_dim,
                    v_dim=v_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    internal_resolution=internal_resolution,
                    block_index=block_index,
                    kernel_size=transformer_kernel_size[block_index],
                )
                for block_index in range(num_layers)
            )
        )
        self.dropout = torch.nn.Dropout(dropout)

        self.learned_positional_bias = torch.nn.Parameter(torch.zeros((1, embed_dim, *internal_resolution)))
        self.periodic_positional_encoding = PositionalEncodingPermute2D(embed_dim)

    def forward(self, x):
        # Apply depthwise separable convolution embedding
        x = self.input_layer_cnn(x)  # (B, D, H, W)
        B, D, H, W = x.shape

        # Add positional embedding (periodic augmented by learned positional bias)
        periodic_positional_encoding = self.periodic_positional_encoding(x)
        learned_positional_bias = self.learned_positional_bias.repeat(B, 1, 1, 1)
        x = x + periodic_positional_encoding + learned_positional_bias

        # Apply Transforrmer
        x = self.dropout(x)
        x = self.transformer(x)
        return x
