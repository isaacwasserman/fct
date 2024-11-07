import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from utils import *


# @torch.compile()
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
        use_attention_bias=True,
        key_projection_stride=1,
        value_projection_stride=1,
    ):
        super().__init__()
        self.q_net = torch.nn.Conv2d(
            embed_dim, q_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.k_net = torch.nn.Conv2d(
            embed_dim,
            q_dim,
            kernel_size=kernel_size,
            stride=key_projection_stride,
            padding=same_padding(kernel_size, format="single"),
        )
        self.v_net = torch.nn.Conv2d(
            embed_dim,
            v_dim,
            kernel_size=kernel_size,
            stride=value_projection_stride,
            padding=same_padding(kernel_size, format="single"),
        )
        if use_attention_bias:
            # self.bias_net = torch.nn.Sequential(
            #     torch.nn.Conv2d(embed_dim, v_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")),
            #     torch.nn.AvgPool2d(kernel_size=internal_resolution),
            # )
            # self.bias_norm = torch.nn.LayerNorm(v_dim)

            # Squeeze and excite bias
            self.bias_net = torch.nn.Linear(embed_dim, v_dim)
        if num_heads > 1:
            self.head_unification = torch.nn.Conv2d(
                v_dim, embed_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
            )
        self.dwc = torch.nn.Conv2d(
            v_dim // num_heads,
            v_dim // num_heads,
            kernel_size=kernel_size,
            padding=same_padding(kernel_size, format="single"),
            groups=v_dim // num_heads,
        )
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.block_index = block_index
        self.internal_resolution = internal_resolution
        self.use_attention_bias = use_attention_bias

    def break_into_heads(self, M):
        B, D, H, W = M.shape
        h = self.num_heads
        return M.reshape(B, h, D // h, H, W).reshape(B * h, D // h, H, W)

    def sum_pool_to_resolution(self, x, output_resolution=3):
        a = torch.nn.functional.interpolate(x, size=(output_resolution, output_resolution), mode="bilinear")
        return a

    def phi(self, x, p=2):
        x = torch.nn.functional.relu(x)
        xp = x**p
        numerator = torch.norm(x) * xp
        denominator = torch.norm(xp)
        return numerator / denominator

    # @torch.compile()
    def spatial_FLatten_attention(self, x):
        Q = self.q_net(x)
        K = self.k_net(x)
        V = self.v_net(x)

        B, Dq, H, W = Q.shape
        _, Dv, _, _ = V.shape
        _, Dm, _, _ = x.shape
        h = self.num_heads

        phi_Q = self.phi(Q)
        phi_K = self.phi(K)

        phi_Q = self.break_into_heads(phi_Q)
        phi_K = self.break_into_heads(phi_K)
        V = self.break_into_heads(V)

        KV = phi_K.unsqueeze(2) * V.unsqueeze(1)
        KV = KV.reshape(-1, *KV.shape[2:])
        KV = self.sum_pool_to_resolution(KV, output_resolution=self.kernel_size)

        abs_max = 32 / (self.kernel_size**2)
        KV = KV.clamp(min=-abs_max, max=abs_max)

        KV = KV.reshape(B * h, Dq // h, Dv // h, self.kernel_size, self.kernel_size)
        KV = KV.permute(0, 2, 1, 3, 4)
        KV = KV.reshape(-1, *KV.shape[2:])

        # Reshape Q into a single B * Dq channel image
        phi_Q = phi_Q.reshape(-1, *phi_Q.shape[2:]).unsqueeze(0)
        bias = None
        if self.use_attention_bias:
            # bias = self.bias_net(x)
            # bias = self.bias_norm(bias.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # bias = bias.reshape(-1)
            # Global pool x to B, Dm, 1, 1

            # Squeeze and excite bias
            squeezed = x.mean(dim=(2, 3))
            excited = self.bias_net(squeezed)
            bias = excited.reshape(-1)

        # QKV is grouped (B*h groups) convolution of Q with KV
        QKV = torch.nn.functional.conv2d(
            phi_Q, KV, bias=bias, groups=B * h, padding=same_padding(KV.shape[-1], format="single")
        )
        QKV = QKV.view(B, Dv, H, W)

        # if QKV.isinf().any():
        #     print("QKV has inf")
        #     print(phi_Q.min().item(), phi_Q.max().item())
        #     print(phi_K.min().item(), phi_K.max().item())
        #     print(V.min().item(), V.max().item())
        #     print(KV.min().item(), KV.max().item())
        #     raise ValueError("QKV has inf")

        # Unify heads
        if h > 1:
            QKV = self.head_unification(QKV)

        # if QKV.isinf().any():
        #     print("QKV has inf")
        #     print(QKV)
        #     raise ValueError("QKV has inf")

        # dwc = self.dwc(V) # (B, Dv, H // value_projection_stride, W)
        # QKV = QKV + self.dwc(V).reshape(B, Dv, H, W)

        # if QKV.isinf().any():
        #     print("QKV has inf")
        #     print(QKV)
        #     raise ValueError("QKV has inf")

        QKV = torch.nn.functional.layer_norm(QKV, QKV.shape[1:])

        return QKV

    # def spatial_linear_self_attention(self, x):
    #     Q = self.q_net(x)
    #     K = self.k_net(x)
    #     V = self.v_net(x)

    #     B, Dq, H, W = Q.shape
    #     _, Dv, _, _ = V.shape
    #     _, Dm, _, _ = x.shape
    #     h = self.num_heads

    #     # Compute softmax of Q over channel dimension
    #     Q = Q - Q.max(dim=-3, keepdim=True)[0]
    #     Q = torch.exp(torch.nn.functional.log_softmax(Q, dim=-3))

    #     # Compute softmax of K over both spatial dimensions simultaneously
    #     K = K.reshape(*K.shape[:2], -1)
    #     K = K - K.max(dim=-1, keepdim=True)[0]
    #     K = torch.exp(torch.nn.functional.log_softmax(K.reshape(*K.shape[:2], -1), dim=-1)).reshape(B, Dq, H, W)

    #     # Break Q, K, V into heads
    #     Q = self.break_into_heads(Q)
    #     K = self.break_into_heads(K)
    #     V = self.break_into_heads(V)

    #     KV = K.unsqueeze(2) * V.unsqueeze(1)
    #     KV = KV.reshape(-1, *KV.shape[2:])
    #     KV = self.sum_pool_to_resolution(KV, output_resolution=self.kernel_size)
    #     KV = KV.reshape(B * h, Dq // h, Dv // h, self.kernel_size, self.kernel_size)
    #     KV = KV.permute(0, 2, 1, 3, 4)
    #     KV = KV.reshape(-1, *KV.shape[2:])

    #     # Reshape Q into a single B * Dq channel image
    #     Q = Q.reshape(-1, *Q.shape[2:]).unsqueeze(0)
    #     bias = None
    #     if self.use_attention_bias:
    #         bias = self.bias_net(x)
    #         bias = self.bias_norm(bias.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    #         bias = bias.reshape(-1)

    #     # QKV is grouped (B*h groups) convolution of Q with KV
    #     QKV = torch.nn.functional.conv2d(Q, KV, bias=bias, groups=B * h, padding=same_padding(KV.shape[-1], format="single"))
    #     QKV = QKV.view(B, Dv, H, W)

    #     # Unify heads
    #     if h > 1:
    #         QKV = self.head_unification(QKV)

    #     return QKV

    def forward(self, x):
        B, D, H, W = x.shape
        # attn = self.spatial_linear_self_attention(x)
        attn = self.spatial_FLatten_attention(x)
        return attn


# Define transformer module
# @torch.compile()
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
        # assert_shape(attn, (B, D, H, W))

        x = x + attn
        # assert_shape(x, (B, D, H, W))

        x = self.layer_norm_2(x)
        # assert_shape(x, (B, D, H, W))

        x = x + self.feed_forward(x)
        # assert_shape(x, (B, D, H, W))
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
