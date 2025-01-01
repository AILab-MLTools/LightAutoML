"""Saint utils."""

from einops import rearrange
from torch import einsum, nn

from ..fttransformer.fttransformer_utils import GEGLU


class Residual(nn.Module):
    """Residual connection layer.

    Args:
        fn : function to apply
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward-pass."""
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """Normalization connection layer.

    Args:
        fn : function to apply
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward-pass."""
        return self.fn(self.norm(x), **kwargs)


# attention


class FeedForward(nn.Module):
    """Feedforward for Transformer block.

    Args:
            dim: Embeddings dimension.
            mult: multiply hidden state dim.
            dropout: Post-Attention dropout.
    """

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult) * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(int(dim * mult), dim)
        )

    def forward(self, x, **kwargs):
        """Forward-pass.

        Args:
            x : torch.Tensor
                3-d tensor; for example, embedded numeric and/or categorical values,
                or the output of a previous attention layer.
            kwargs: kwargs

        Returns:
            torch.Tensor

        """
        return self.net(x, **kwargs)


class Attention(nn.Module):
    """Attention Block.

    Args:
            dim: Embeddings dimension.
            heads: Number of heads in Attention.
            dim_head: Attention head dimension.
            dropout: Post-Attention dropout.
    """

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Transform the input tensor with attention.

        Args:
            x : torch.Tensor
                3-d tensor; for example, embedded numeric and/or categorical values,
                or the output of a previous attention layer.
            mask: torch.Tensor

        Returns:
            torch.Tensor

        """
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            sim[~mask[None, None].expand_as(sim)] = float("-inf")
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    """Transformer Block.

    Args:
            dim: Embeddings dimension.
            nfeats: Number of features.
            depth: Number of Attention Blocks.
            heads: Number of heads in Attention.
            dim_head: Attention head dimension.
            ffn_mult: multiply hidden state of feed forward layer.
            attn_dropout: Post-Attention dropout.
            ff_dropout: Feed-Forward Dropout.
            style: attention style: 'col' or 'colrow'
    """

    def __init__(self, dim, nfeats, depth, heads, dim_head, ffn_mult, attn_dropout, ff_dropout, style="col"):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == "colrow":
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))
                            ),
                            PreNorm(dim, Residual(FeedForward(dim, mult=ffn_mult, dropout=ff_dropout))),
                            PreNorm(
                                dim * nfeats,
                                Residual(Attention(dim * nfeats, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                            ),
                            PreNorm(
                                dim * nfeats, Residual(FeedForward(dim * nfeats, mult=ffn_mult, dropout=ff_dropout))
                            ),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim * nfeats,
                                Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout)),
                            ),
                            PreNorm(
                                dim * nfeats, Residual(FeedForward(dim * nfeats, mult=ffn_mult, dropout=ff_dropout))
                            ),
                        ]
                    )
                )

    def forward(self, x, mask_features=None, mask_samples=None):
        """Transform the input embeddings tensor with Transformer module.

        Args:
            x : torch.Tensor
                3-d tensor; embedded numeric and/or categorical values,
                or the output of a previous Transformer layer.
            mask_features: torch.Tensor
                mask for the first attention
            mask_samples: torch.Tensor
                mask for the second attention

        Returns:
            torch.Tensor

        """
        _, n, _ = x.shape
        if self.style == "colrow":
            for attn1, ff1, attn2, ff2 in self.layers:  # type: ignore[code]
                x = attn1(x, mask=mask_features)
                x = ff1(x)
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn2(x, mask=mask_samples)
                x = ff2(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        else:
            for attn1, ff1 in self.layers:  # type: ignore[code]
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        return x


# transformer
class ColTransformer(nn.Module):
    """Transformer Block.

    Args:
            dim: Embeddings dimension.
            depth: Number of Attention Blocks.
            heads: Number of heads in Attention.
            dim_head: Attention head dimension.
            attn_dropout: Post-Attention dropout.
            ff_dropout: Feed-Forward Dropout.
    """

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                        PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]
                )
            )

    def forward(self, x, mask_features=None, mask_samples=None):
        """Transform the input embeddings tensor with Transformer module.

        Args:
            x : torch.Tensor
                3-d tensor; embedded numeric and/or categorical values,
                or the output of a previous Transformer layer.
            mask_features: torch.Tensor
                not used
            mask_samples: torch.Tensor
                not used

        Returns:
            torch.Tensor

        """
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
