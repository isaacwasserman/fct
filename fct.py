import torch
from utils import *


# Define transformer module
class TransformerBlock(torch.nn.Module):
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
        self.q_net = torch.nn.Conv2d(
            embed_dim, q_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.k_net = torch.nn.Conv2d(
            embed_dim, q_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.v_net = torch.nn.Conv2d(
            embed_dim, v_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
        )
        self.head_unification = torch.nn.Conv2d(
            v_dim, embed_dim, kernel_size=kernel_size, padding=same_padding(kernel_size, format="single")
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

    def break_into_heads(self, M):
        B, D, H, W = M.shape
        h = self.num_heads
        return M.reshape(B, h, D // h, H, W).flatten(0, 1)

    def spatial_linear_self_attention(self, Q, K, V):
        """
        Q: (B, Dq, H, W)
        K: (B, Dq, H, W)
        V: (B, Dv, H, W)
        """

        B, Dq, H, W = Q.shape
        _, Dv, _, _ = V.shape
        h = self.num_heads

        assert_shape(Q, (B, Dq, H, W))
        assert_shape(K, (B, Dq, H, W))
        assert_shape(V, (B, Dv, H, W))

        # Compute softmax of Q over channel dimension
        Q = torch.nn.functional.softmax(Q, dim=-3)
        assert_shape(Q, (B, Dq, H, W))

        # Compute softmax of K over both spatial dimensions simultaneously
        K = torch.nn.functional.softmax(K.flatten(-2), dim=-1).reshape(B, Dq, H, W)
        assert_shape(K, (B, Dq, H, W))

        # Break Q, K, V into heads
        Q = self.break_into_heads(Q)
        assert_shape(Q, (B * h, Dq // h, H, W))
        K = self.break_into_heads(K)
        assert_shape(K, (B * h, Dq // h, H, W))
        V = self.break_into_heads(V)
        assert_shape(V, (B * h, Dv // h, H, W))

        # Compute K^T @ V (kind of)
        # i.e. outer product of K and V, summed over spatial dimensions
        # (Bh, Dq//h, 1, H, W) x (Bh, 1, Dv//h, H, W) -> (Bh, Dq//h, Dv//h, H, W) -> (Bh, Dq//h, Dv//h)
        KV = (K.unsqueeze(2) * V.unsqueeze(1)).sum(dim=[-1, -2])
        assert_shape(KV, (B * h, Dq // h, Dv // h))

        # Reshape KV into 2D convolutional kernels
        KV = KV.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 1, 3, 4).flatten(0, 1)
        assert_shape(KV, (B * h * (Dv // h), Dq // h, 1, 1))

        # Reshape Q into a single B * Dq channel image
        Q = Q.flatten(0, 1).unsqueeze(0)
        assert_shape(Q, (1, B * h * (Dq // h), H, W))

        # QKV is grouped (B*h groups) convolution of Q with KV
        QKV = torch.nn.functional.conv2d(Q, KV, groups=B * h)
        assert_shape(QKV, (1, B * h * Dv // h, H, W))
        QKV = QKV.view(B, Dv, H, W)
        assert_shape(QKV, (B, Dv, H, W))

        # Unify heads
        QKV = self.head_unification(QKV)

        return QKV

    def spatial_quadratic_self_attention(self, Q, K, V):
        B, Dq, H, W = Q.shape
        _, Dv, _, _ = V.shape
        h = self.num_heads

        assert_shape(Q, (B, Dq, H, W))
        assert_shape(K, (B, Dq, H, W))
        assert_shape(V, (B, Dv, H, W))

        Q = self.break_into_heads(Q)
        assert_shape(Q, (B * h, Dq // h, H, W))

        K = self.break_into_heads(K)
        assert_shape(K, (B * h, Dq // h, H, W))

        V = self.break_into_heads(V)
        assert_shape(V, (B * h, Dv // h, H, W))

        Q_image = Q.flatten(0, 1).unsqueeze(0)
        assert_shape(Q_image, (1, B * h * (Dq // h), H, W))

        K_kernels = K.flatten(-2).flatten(0, 1).unsqueeze(-1).unsqueeze(-1)
        assert_shape(K_kernels, (B * h * (Dq // h), H * W, 1, 1))
        K_kernels = K_kernels.permute(1, 0, 2, 3)
        assert_shape(K_kernels, (H * W, B * h * (Dq // h), 1, 1))
        K_kernels = K_kernels.reshape(H * W, B * h, Dq // h, 1, 1).permute(1, 0, 2, 3, 4)
        assert_shape(K_kernels, (B * h, H * W, Dq // h, 1, 1))
        K_kernels = K_kernels.flatten(0, 1)
        assert_shape(K_kernels, (B * h * H * W, Dq // h, 1, 1))

        QK = torch.nn.functional.conv2d(Q_image, K_kernels, groups=B * h)
        assert_shape(QK, (1, B * h * H * W, H, W))
        QK = QK.squeeze(0).reshape(B * h, H * W, H, W).flatten(-2)
        QK = torch.nn.functional.softmax(QK, dim=-1)
        assert_shape(QK, (B * h, H * W, H * W))
        QK = QK.reshape(B * h, H * W, H, W)
        QKV = (QK.unsqueeze(2) * V.unsqueeze(1)).sum(
            dim=[-1, -2]
        )  # (Bh, HW, 1, H, W) x (Bh, 1, Dv//h, H, W) -> (Bh, HW, Dv // h, H, W) -> (Bh, HW, Dv // h)
        assert_shape(QKV, (B * h, H * W, Dv // h))
        QKV = QKV.permute(0, 2, 1).reshape(B * h, Dv // h, H, W)
        assert_shape(QKV, (B * h, Dv // h, H, W))
        QKV = QKV.reshape(B, Dv, H, W)
        assert_shape(QKV, (B, Dv, H, W))
        QKV = self.head_unification(QKV)
        assert_shape(QKV, (B, self.embed_dim, H, W))

        return QKV

    def forward(self, x):
        B, D, H, W = x.shape
        Dq = self.q_dim
        Dv = self.v_dim
        after_norm_1 = self.layer_norm_1(x)
        assert_shape(after_norm_1, (B, D, H, W))
        Q = self.q_net(after_norm_1)
        assert_shape(Q, (B, Dq, H, W))
        K = self.k_net(after_norm_1)
        assert_shape(K, (B, Dq, H, W))
        V = self.v_net(after_norm_1)
        assert_shape(V, (B, Dv, H, W))

        attn = self.spatial_linear_self_attention(Q, K, V)
        assert_shape(attn, (B, D, H, W))
        x = x + attn
        assert_shape(x, (B, D, H, W))
        x = self.layer_norm_2(x)
        assert_shape(x, (B, D, H, W))
        x = x + self.feed_forward(x)
        assert_shape(x, (B, D, H, W))
        return x
