"""
modules.py - GFNet (Global Filter Network) components in pure PyTorch.

Follows the architecture of:
    Rao, Zhao, Zhu, Lu, Zhou. "Global Filter Networks for Image Classification".
    NeurIPS 2021. arXiv:2107.00645

Two families are provided, matching the paper:

    GFNet         isotropic (ViT/DeiT-style) model: patch embed -> N x Block -> norm -> GAP -> head
                  variants: gfnet_ti, gfnet_xs, gfnet_s, gfnet_b                     (paper Table 3)
    GFNetPyramid  hierarchical model with 4 stages and downsampling between them
                  variants: gfnet_h_ti, gfnet_h_s, gfnet_h_b                         (paper Table 4)

A standard convolutional baseline is also provided for comparison:

    BaselineCNN   VGG-style CNN: 4 stages of [Conv3x3-BN-ReLU] x 2 + MaxPool,
                  global average pooling, linear head                              ("cnn")

All models expose forward_tokens(x) -> (B, N, D) final spatial features before
pooling, and token_grid = (h, w), so predict.py can draw the same evidence
heatmaps for every model (head is linear on the average of these features).

Each GFNet block is

    x <- x + DropPath( MLP( LN( GlobalFilter( LN(x) ) ) ) )

where GlobalFilter performs token mixing in the frequency domain:

    X = rfft2(x)          2D FFT over the token grid, per channel
    X = X * K             K: learnable complex filter, shape (h, w//2 + 1, dim)
    x = irfft2(X)

Only PyTorch is used (no NumPy, timm or other third-party libraries).
Defaults for the ADNI task: in_chans=1 (greyscale MRI), num_classes=2 (NC=0, AD=1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(x):
    """Return x as an (h, w) tuple."""
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


# --------------------------------------------------------------------------- #
# Regularisation
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    """Stochastic depth: randomly drop the whole residual branch per sample.

    Kept residual outputs are rescaled by 1 / keep_prob so the expected value
    is unchanged. Identity at evaluation time.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)          # one draw per sample
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


# --------------------------------------------------------------------------- #
# Channel mixing
# --------------------------------------------------------------------------- #
class Mlp(nn.Module):
    """Feed-forward network applied independently to every token (channel mixing).

    Linear(dim -> hidden) -> GELU -> Dropout -> Linear(hidden -> dim) -> Dropout
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# --------------------------------------------------------------------------- #
# Token mixing: the global filter
# --------------------------------------------------------------------------- #
class GlobalFilter(nn.Module):
    """Global filter layer (paper Sec. 3.2, Eq. 4-5).

    Mixes information across all tokens, separately for each channel, by an
    element-wise product with a learnable complex filter in the 2D Fourier domain.
    Equivalent to a depthwise circular convolution whose kernel spans the whole
    token grid, at O(N log N) cost.

    Args:
        dim:       channel dimension D
        grid_size: token grid (h, w), e.g. (14, 14) for 224 px input with 16 px patches

    Parameter:
        complex_weight: (h, w//2 + 1, dim, 2) real tensor, last axis = (real, imag).
                        Initialised N(0, 0.02^2) as in the official code.
    """

    def __init__(self, dim: int, grid_size=(14, 14)):
        super().__init__()
        self.h, self.w = _pair(grid_size)
        self.freq_w = self.w // 2 + 1        # rfft2 keeps only half the spectrum
        self.complex_weight = nn.Parameter(
            torch.randn(self.h, self.freq_w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if N != self.h * self.w:
            raise ValueError(f"Expected {self.h * self.w} tokens, got {N}")

        # cuFFT in half precision only supports power-of-two sizes, so the FFT
        # always runs in float32 even under mixed-precision training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.reshape(B, self.h, self.w, C).to(torch.float32)
            x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")              # (B, h, w//2+1, C)
            x = x * torch.view_as_complex(self.complex_weight)
            x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm="ortho")
        return x.reshape(B, N, C)

    def extra_repr(self) -> str:
        return f"grid=({self.h}, {self.w}), freq=({self.h}, {self.freq_w})"


# --------------------------------------------------------------------------- #
# GFNet blocks
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """GFNet block (paper Fig. 1):

        x <- x + DropPath( MLP( LN2( GlobalFilter( LN1(x) ) ) ) )

    A single residual connection wraps both the filter and the MLP.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0,
                 drop_path: float = 0.0, act_layer=nn.GELU, norm_eps: float = 1e-6,
                 grid_size=(14, 14)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.filter = GlobalFilter(dim, grid_size=grid_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))


class BlockLayerScale(Block):
    """GFNet block with LayerScale (used by the hierarchical models):

        x <- x + DropPath( gamma * MLP( LN2( GlobalFilter( LN1(x) ) ) ) )

    gamma is a learnable per-channel scale initialised to a small value, which
    stabilises training of deeper networks.
    """

    def __init__(self, dim: int, init_values: float = 1e-5, **kwargs):
        super().__init__(dim, **kwargs)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(
            self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))


# --------------------------------------------------------------------------- #
# Tokenisation and downsampling
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """Split the image into non-overlapping patches and linearly project each
    one to `embed_dim` channels. Implemented as Conv2d(kernel=stride=patch).

    (B, C_in, H, W) -> (B, N, D), with N = (H / p) * (W / p)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans: int = 1,
                 embed_dim: int = 384):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        if any(s % p for s, p in zip(self.img_size, self.patch_size)):
            raise ValueError(f"img_size {self.img_size} not divisible by "
                             f"patch_size {self.patch_size}")
        self.grid_size = (self.img_size[0] // self.patch_size[0],
                          self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Input size ({H}x{W}) does not match model "
                             f"({self.img_size[0]}x{self.img_size[1]})")
        return self.proj(x).flatten(2).transpose(1, 2)        # (B, N, D)


class DownLayer(nn.Module):
    """Downsampling between hierarchical stages: a 2x2 stride-2 convolution
    that halves the token grid and changes the channel width.

    (B, h*w, D_in) -> (B, (h/2)*(w/2), D_out)
    """

    def __init__(self, grid_size=(56, 56), dim_in: int = 64, dim_out: int = 128):
        super().__init__()
        self.h, self.w = _pair(grid_size)
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = (self.h // 2) * (self.w // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(B, self.h, self.w, C).permute(0, 3, 1, 2)   # (B, C, h, w)
        x = self.proj(x)                                           # (B, D_out, h/2, w/2)
        return x.flatten(2).transpose(1, 2)                        # (B, N/4, D_out)


# --------------------------------------------------------------------------- #
# Weight initialisation (as in the official code)
# --------------------------------------------------------------------------- #
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# Isotropic GFNet
# --------------------------------------------------------------------------- #
class GFNet(nn.Module):
    """Isotropic GFNet (paper Table 3).

    patch embed + learned positional embedding -> depth x Block -> LayerNorm
    -> global average pool over tokens -> (dropout) -> linear head

    Args:
        img_size:       input size, int or (H, W)
        patch_size:     patch size p
        in_chans:       input channels (1 for greyscale MRI)
        num_classes:    output logits (2 for AD vs NC)
        embed_dim:      channel width D
        depth:          number of GFNet blocks
        mlp_ratio:      MLP hidden width = mlp_ratio * D
        drop_rate:      dropout inside MLPs and after the positional embedding
        drop_path_rate: maximum stochastic-depth rate
        uniform_drop:   same drop-path rate for every block (else linear 0 -> max)
        dropcls:        dropout before the classifier head
    """

    def __init__(self, img_size=224, patch_size=16, in_chans: int = 1,
                 num_classes: int = 2, embed_dim: int = 384, depth: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0, uniform_drop: bool = False,
                 dropcls: float = 0.0, norm_eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid = self.patch_embed.grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        if uniform_drop:
            dpr = [drop_path_rate] * depth
        else:
            dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList([
            Block(embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                  norm_eps=norm_eps, grid_size=grid)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.final_dropout = nn.Dropout(dropcls) if dropcls > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights)

    def no_weight_decay(self):
        """Parameter names to exclude from weight decay (as in the official code)."""
        return {"pos_embed"}

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @property
    def token_grid(self):
        return self.patch_embed.grid_size

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> normalised final tokens (B, N, D), before pooling."""
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> pooled features (B, D)."""
        return self.forward_tokens(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> logits (B, num_classes). Softmax is NOT applied."""
        return self.head(self.final_dropout(self.forward_features(x)))


# --------------------------------------------------------------------------- #
# Hierarchical GFNet
# --------------------------------------------------------------------------- #
class GFNetPyramid(nn.Module):
    """Hierarchical GFNet (paper Table 4).

    4 stages; stage 1 uses 4x4 patches (56x56 grid at 224 px), and a 2x2
    stride-2 DownLayer halves the grid between stages (56 -> 28 -> 14 -> 7).
    Blocks use LayerScale unless no_layerscale=True.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans: int = 1,
                 num_classes: int = 2, embed_dim=(64, 128, 256, 512),
                 depth=(2, 2, 10, 4), mlp_ratio=(4, 4, 4, 4), drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0, init_values: float = 0.001,
                 no_layerscale: bool = False, dropcls: float = 0.0,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = list(embed_dim)
        self.num_features = self.embed_dim[-1]
        n_stages = len(self.embed_dim)

        # stage 0: patch embedding; stages 1..: downsampling
        first = PatchEmbed(img_size, patch_size, in_chans, self.embed_dim[0])
        grids = [first.grid_size]
        for _ in range(n_stages - 1):
            h, w = grids[-1]
            if h % 2 or w % 2:
                raise ValueError(f"Token grid {grids[-1]} cannot be halved")
            grids.append((h // 2, w // 2))

        self.patch_embed = nn.ModuleList([first] + [
            DownLayer(grids[i], self.embed_dim[i], self.embed_dim[i + 1])
            for i in range(n_stages - 1)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, first.num_patches, self.embed_dim[0]))
        self.pos_drop = nn.Dropout(drop_rate)
        self.token_grid = grids[-1]

        dpr = torch.linspace(0, drop_path_rate, sum(depth)).tolist()
        self.blocks = nn.ModuleList()
        cur = 0
        for i in range(n_stages):
            common = dict(mlp_ratio=mlp_ratio[i], drop=drop_rate,
                          norm_eps=norm_eps, grid_size=grids[i])
            if no_layerscale:
                stage = [Block(self.embed_dim[i], drop_path=dpr[cur + j], **common)
                         for j in range(depth[i])]
            else:
                stage = [BlockLayerScale(self.embed_dim[i], init_values=init_values,
                                         drop_path=dpr[cur + j], **common)
                         for j in range(depth[i])]
            self.blocks.append(nn.Sequential(*stage))
            cur += depth[i]

        self.norm = nn.LayerNorm(self.num_features, eps=norm_eps)
        self.final_dropout = nn.Dropout(dropcls) if dropcls > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights)

    def no_weight_decay(self):
        return {"pos_embed"}

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> normalised last-stage tokens (B, N, D), before pooling."""
        for i, (embed, stage) in enumerate(zip(self.patch_embed, self.blocks)):
            x = embed(x)
            if i == 0:
                x = self.pos_drop(x + self.pos_embed)
            x = stage(x)
        return self.norm(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.final_dropout(self.forward_features(x)))


# --------------------------------------------------------------------------- #
# Baseline: standard convolutional network
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """[Conv3x3 -> BatchNorm -> ReLU] x 2, then 2x2 max-pooling (halves H and W)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class BaselineCNN(nn.Module):
    """Standard VGG-style CNN used as the baseline for GFNet.

    4 ConvBlocks with widths (32, 64, 128, 256): 224 -> 112 -> 56 -> 28 -> 14,
    so the final feature map is a 14 x 14 grid, the same as GFNet's token grid.
    Global average pooling -> dropout -> linear head. ~1.2M parameters.

    Only local 3x3 convolutions mix spatial information, so its receptive field
    grows gradually with depth, in contrast to GFNet's global filters.
    """

    def __init__(self, img_size=224, in_chans: int = 1, num_classes: int = 2,
                 widths=(32, 64, 128, 256), drop_rate: float = 0.0, **kwargs):
        # **kwargs absorbs GFNet-only options (e.g. drop_path_rate) passed by train.py
        super().__init__()
        self.num_classes = num_classes
        self.num_features = widths[-1]
        chans = (in_chans,) + tuple(widths)
        self.features = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1])
                                        for i in range(len(widths))])
        h, w = _pair(img_size)
        scale = 2 ** len(widths)
        if h % scale or w % scale:
            raise ValueError(f"img_size {(h, w)} must be divisible by {scale}")
        self.token_grid = (h // scale, w // scale)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def no_weight_decay(self):
        return set()

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> final feature map as tokens (B, N, D), before pooling."""
        return self.features(x).flatten(2).transpose(1, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> logits (B, num_classes). Softmax is NOT applied."""
        return self.head(self.dropout(self.forward_features(x)))


def baseline_cnn(**kwargs) -> BaselineCNN:
    """Standard 8-layer VGG-style CNN baseline (~1.2M params)."""
    return BaselineCNN(**kwargs)


# --------------------------------------------------------------------------- #
# Model variants (configurations from the official repository)
# --------------------------------------------------------------------------- #
def gfnet_ti(**kwargs) -> GFNet:
    """GFNet-Ti: D=256, 12 blocks (~7M params, 1.3 GFLOPs at 224 px RGB)."""
    return GFNet(patch_size=16, embed_dim=256, depth=12, mlp_ratio=4, **kwargs)


def gfnet_xs(**kwargs) -> GFNet:
    """GFNet-XS: D=384, 12 blocks (~16M params, 2.9 GFLOPs)."""
    return GFNet(patch_size=16, embed_dim=384, depth=12, mlp_ratio=4, **kwargs)


def gfnet_s(**kwargs) -> GFNet:
    """GFNet-S: D=384, 19 blocks (~25M params, 4.5 GFLOPs)."""
    return GFNet(patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, **kwargs)


def gfnet_b(**kwargs) -> GFNet:
    """GFNet-B: D=512, 19 blocks (~43M params, 7.9 GFLOPs)."""
    return GFNet(patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, **kwargs)


def gfnet_h_ti(**kwargs) -> GFNetPyramid:
    """GFNet-H-Ti (~15M params, 2.1 GFLOPs)."""
    kwargs.setdefault("drop_path_rate", 0.1)
    return GFNetPyramid(patch_size=4, embed_dim=(64, 128, 256, 512),
                        depth=(3, 3, 10, 3), mlp_ratio=(4, 4, 4, 4), **kwargs)


def gfnet_h_s(**kwargs) -> GFNetPyramid:
    """GFNet-H-S (~32M params, 4.6 GFLOPs)."""
    kwargs.setdefault("drop_path_rate", 0.2)
    kwargs.setdefault("init_values", 1e-5)
    return GFNetPyramid(patch_size=4, embed_dim=(96, 192, 384, 768),
                        depth=(3, 3, 10, 3), mlp_ratio=(4, 4, 4, 4), **kwargs)


def gfnet_h_b(**kwargs) -> GFNetPyramid:
    """GFNet-H-B (~54M params, 8.6 GFLOPs)."""
    kwargs.setdefault("drop_path_rate", 0.4)
    kwargs.setdefault("init_values", 1e-6)
    return GFNetPyramid(patch_size=4, embed_dim=(96, 192, 384, 768),
                        depth=(3, 3, 27, 3), mlp_ratio=(4, 4, 4, 4), **kwargs)


MODELS = {
    "gfnet-ti": gfnet_ti, "gfnet-xs": gfnet_xs, "gfnet-s": gfnet_s, "gfnet-b": gfnet_b,
    "gfnet-h-ti": gfnet_h_ti, "gfnet-h-s": gfnet_h_s, "gfnet-h-b": gfnet_h_b,
    "cnn": baseline_cnn,
}


def build_model(name: str = "gfnet-xs", **kwargs) -> nn.Module:
    """Build a variant by name, e.g. build_model("gfnet-xs", drop_path_rate=0.1)."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


# --------------------------------------------------------------------------- #
# Loading ImageNet checkpoints from the official repo
# --------------------------------------------------------------------------- #
def adapt_pretrained_state_dict(state_dict: dict, model: nn.Module) -> dict:
    """Make an official GFNet ImageNet checkpoint loadable into `model`.

    Handles:
      * RGB -> greyscale: patch-embed weights summed over the 3 input channels
      * different token grid: bicubic interpolation of pos_embed and of every
        global filter (real and imaginary parts interpolated separately)
      * different number of classes: the classifier head is dropped
    Load the result with model.load_state_dict(sd, strict=False).
    """
    sd = dict(state_dict.get("model", state_dict))
    target = model.state_dict()

    for k in list(sd):
        if k not in target:
            sd.pop(k)
            continue
        src, dst = sd[k], target[k]
        if src.shape == dst.shape:
            continue

        if k.endswith("proj.weight") and src.ndim == 4 and src.shape[1] != dst.shape[1]:
            # (D, 3, p, p) -> (D, 1, p, p)
            sd[k] = src.sum(dim=1, keepdim=True) if dst.shape[1] == 1 else \
                src.mean(dim=1, keepdim=True).repeat(1, dst.shape[1], 1, 1)
        elif k == "pos_embed":
            # (1, N_src, D) -> (1, N_dst, D) on the 2D token grid
            n_src = src.shape[1]
            s = int(round(n_src ** 0.5))
            grid = model.patch_embed[0].grid_size if isinstance(model.patch_embed, nn.ModuleList) \
                else model.patch_embed.grid_size
            pe = src.reshape(1, s, s, -1).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=grid, mode="bicubic", align_corners=False)
            sd[k] = pe.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1)
        elif k.endswith("complex_weight"):
            # (h_src, w_src, D, 2) -> (h_dst, w_dst, D, 2)
            h, w = dst.shape[:2]
            cw = src.permute(2, 3, 0, 1).reshape(1, -1, src.shape[0], src.shape[1])
            cw = F.interpolate(cw, size=(h, w), mode="bicubic", align_corners=False)
            sd[k] = cw.reshape(src.shape[2], 2, h, w).permute(2, 3, 0, 1).contiguous()
        else:
            sd.pop(k)   # e.g. head.weight / head.bias with a different num_classes
    return sd
