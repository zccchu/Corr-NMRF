"""Corr-NMRF message passing with epipolar-aware hidden-state correction."""

import math
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import Mlp, DropPath


class LowRankTTTLearner(nn.Module):
    def __init__(self, num_heads, head_dim, rank=8, use_diag=True):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_heads, head_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(num_heads, head_dim, rank) * 0.02)
        if use_diag:
            self.diag = nn.Parameter(torch.ones(num_heads, head_dim))
        else:
            self.register_parameter("diag", None)
        self.bias = nn.Parameter(torch.zeros(1, 1, num_heads, head_dim))

    def forward(self, x):
        low = torch.einsum("bnhd,hdr->bnhr", x, self.A)
        low = torch.einsum("bnhr,hdr->bnhd", low, self.B)
        if self.diag is not None:
            low = low + x * self.diag[None, None, :, :]
        return low + self.bias


class SpatialStateFusion(nn.Module):
    def __init__(self, dim, kernels=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    groups=dim,
                    bias=True,
                )
                for d in kernels
            ]
        )
        self.alpha = nn.Parameter(torch.ones(len(kernels)))

    def forward(self, x):
        weights = torch.softmax(self.alpha, dim=0)
        out = 0.0
        for w, conv in zip(weights, self.convs):
            out = out + w * conv(x)
        return out


class StereoContextProjector(nn.Module):
    def __init__(self, context_dim, out_dim, cost_group):
        super().__init__()
        self.left_proj = nn.Linear(context_dim, out_dim)
        self.right_proj = nn.Linear(context_dim, out_dim)
        self.corr_proj = nn.Linear(cost_group, out_dim)

    def forward(self, left_feat, right_feat, corr):
        left_embed = self.left_proj(left_feat)
        right_embed = self.right_proj(right_feat)
        corr_embed = self.corr_proj(corr)
        return left_embed, right_embed, corr_embed


class HardTokenRouterPerPixel(nn.Module):
    """Top-k along candidate dimension N independently for each spatial token (each row of [BHW, N])."""

    def __init__(self, hard_token_ratio=0.5):
        super().__init__()
        self.hard_token_ratio = hard_token_ratio

    def forward(self, loss_token):
        # loss_token: [BHW, N]
        score = loss_token
        BHW, N = score.shape
        if self.hard_token_ratio >= 1.0:
            return torch.ones(BHW, N, device=score.device, dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)

        k = min(N, max(1, int(round(N * self.hard_token_ratio))))
        topk_vals, _ = torch.topk(score, k, dim=1, largest=True)
        threshold = topk_vals[:, -1:]  # [BHW, 1]
        mask = score >= threshold  # [BHW, N]
        return mask.unsqueeze(-1).unsqueeze(-1)


class CorrNMRFNMP(nn.Module):
    """Epipolar target: ``right_feat`` is already disparity-warped per candidate (see NMP Inference).

    - ``split_ln``: T = LN(E_R - E_L) + g LN(E_C), where g is a learnable candidate gate.
    - ``sum_ln``: T = LN(E_R - E_L + E_C).

    Loss / update: ``l2`` uses err and MSE token score; ``smooth_l1`` uses Huber-style token score and
    derivative-shaped update direction (one quasi-gradient step on Smooth L1).
    """

    def __init__(
        self,
        dim,
        qkv_dim,
        num_heads=4,
        rank=8,
        mini_batch_size=16,
        max_chunks=32,
        base_lr=1.0,
        hard_token_ratio=0.5,
        use_spatial_fusion=True,
        fusion_kernels=(1, 3, 5),
        dropout=0.0,
        mlp_ratio=4.0,
        normalize_before=True,
        target_mode="epipolar",
        stop_grad_target=True,
        use_confidence_weight=True,
        use_diag=True,
        drop_path=0.0,
        cost_group=32,
        context_dim=64,
        use_residual=True,
        epipolar_target_style="split_ln",
        corr_lambda=0.5,
        ttt_loss_type="smooth_l1",
        smooth_l1_beta=1.0,
        router_loss_weight_conf=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.dim = dim
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank
        self.mini_batch_size = mini_batch_size
        self.max_chunks = max_chunks
        self.base_lr = base_lr
        self.hard_token_ratio = hard_token_ratio
        self.use_spatial_fusion = use_spatial_fusion
        self.normalize_before = normalize_before
        self.target_mode = target_mode
        self.stop_grad_target = stop_grad_target
        self.use_confidence_weight = use_confidence_weight
        self.use_diag = use_diag
        self.use_residual = use_residual

        self.H = None
        self.W = None

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(qkv_dim, dim)
        self.k_proj = nn.Linear(qkv_dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.lr_gate = nn.Linear(qkv_dim, num_heads)
        self.context_proj = StereoContextProjector(context_dim=context_dim, out_dim=dim, cost_group=cost_group)
        self.corr_conf = nn.Sequential(
            nn.Linear(cost_group, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.learner = LowRankTTTLearner(
            num_heads=num_heads,
            head_dim=self.head_dim,
            rank=rank,
            use_diag=use_diag,
        )
        if use_spatial_fusion:
            self.spatial_fusion = SpatialStateFusion(dim, kernels=fusion_kernels)
        else:
            self.spatial_fusion = None
        self.o_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=dropout,
        )

        self.epipolar_target_style = epipolar_target_style
        self.corr_lambda = float(corr_lambda)
        self.ttt_loss_type = ttt_loss_type
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.router_loss_weight_conf = bool(router_loss_weight_conf)

        self.msg_gamma = nn.Parameter(torch.zeros(1))
        self.target_norm = nn.LayerNorm(self.dim)
        if epipolar_target_style == "split_ln":
            self.ln_diff = nn.LayerNorm(self.dim)
            self.ln_corr = nn.LayerNorm(self.dim)
            self.corr_gate = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, 1),
            )
            init_gate = min(max(self.corr_lambda, 1e-4), 1.0 - 1e-4)
            nn.init.zeros_(self.corr_gate[-1].weight)
            nn.init.constant_(self.corr_gate[-1].bias, math.log(init_gate / (1.0 - init_gate)))
        else:
            self.ln_diff = None
            self.ln_corr = None
            self.corr_gate = None

        self.router = HardTokenRouterPerPixel(self.hard_token_ratio)

    def _ensure_corr_3d(self, corr, BHW, N, device, dtype):
        """Align stereo ``corr`` to ``[BHW, N, G]`` for context_proj / corr_conf (handles flat cost rows)."""
        G = self.context_proj.corr_proj.in_features
        if corr is None:
            return torch.zeros(BHW, N, G, device=device, dtype=dtype)
        if corr.dim() == 3:
            br, bn, bg = corr.shape
            if bg != G:
                raise ValueError(f"corr last dim {bg} != cost_group {G}")
            if br == BHW and bn == N:
                return corr
            if br * bn == BHW * N:
                return corr.reshape(BHW, N, G)
            if br == BHW:
                if bn > N:
                    return corr[:, :N, :].contiguous()
                if bn < N:
                    pad = torch.zeros(BHW, N - bn, G, device=device, dtype=dtype)
                    return torch.cat([corr, pad], dim=1)
            raise ValueError(
                f"corr 3D shape {tuple(corr.shape)} incompatible with label_rep ({BHW},{N},.)"
            )
        if corr.dim() == 2:
            nrow, ncol = corr.shape
            if ncol != G:
                raise ValueError(f"corr 2D last dim {ncol} != cost_group {G}")
            if nrow == BHW * N:
                return corr.view(BHW, N, G)
            if nrow % BHW == 0:
                bn = nrow // BHW
                c3 = corr.view(BHW, bn, G)
                if bn == N:
                    return c3
                if bn > N:
                    return c3[:, :N, :].contiguous()
                pad = torch.zeros(BHW, N - bn, G, device=device, dtype=dtype)
                return torch.cat([c3, pad], dim=1)
        raise ValueError(f"corr must be 2D or 3D, got shape {tuple(corr.shape)}")

    def _update_dir_and_loss_token(self, err):
        """err: [BHW,N,H,D]. loss_token: [BHW,N] (mean over heads & D); update_dir same shape as err."""
        if self.ttt_loss_type == "l2":
            loss_token = err.pow(2).mean(dim=(2, 3))
        else:
            beta = self.smooth_l1_beta
            abs_e = err.abs()
            sl1_elem = torch.where(abs_e < beta, 0.5 * (err**2) / beta, abs_e - 0.5 * beta)
            loss_token = sl1_elem.mean(dim=(2, 3))
        update_dir = err
        if self.ttt_loss_type != "l2":
            abs_e = err.abs()
            beta = self.smooth_l1_beta
            update_dir = torch.where(abs_e < beta, err / beta, torch.sign(err))
        return loss_token, update_dir

    def _confidence_from_corr(self, corr, BHW, N):
        """Return ``conf`` of shape ``[BHW, N]``. Always re-align ``corr`` (same rules as ``_ensure_corr_3d``)
        so ``corr_conf`` never sees a wrong ``[BHW, huge, G]`` layout (would yield conf ``[BHW, huge]``)."""
        corr = self._ensure_corr_3d(corr, BHW, N, corr.device, corr.dtype)
        logits = self.corr_conf(corr)
        return torch.sigmoid(logits[..., 0])

    def _forward_impl(self, label_rep, abs_encoding, stereo_context=None):
        assert self.H is not None and self.W is not None
        BHW, N, C = label_rep.shape
        H, W = self.H, self.W
        B = BHW // (H * W)

        shortcut = label_rep
        x = self.norm1(label_rep) if self.normalize_before else label_rep
        qk_input = torch.cat([x, abs_encoding], dim=-1)

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(x)

        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)

        qk_state = 0.5 * (q + k)
        qk_state = qk_state + v * 0.0
        mb = max(1, int(self.mini_batch_size))
        max_chunks = int(self.max_chunks)
        if max_chunks > 0 and BHW > mb and ((BHW + mb - 1) // mb) > max_chunks:
            mb = (BHW + max_chunks - 1) // max_chunks
        if mb < BHW:
            pred_chunks = []
            eta_chunks = []
            for start in range(0, BHW, mb):
                end = min(BHW, start + mb)
                qk_chunk_state = qk_state[start:end]
                qk_chunk = qk_input[start:end]
                pred_chunks.append(self.learner(qk_chunk_state))
                eta_chunks.append(torch.sigmoid(self.lr_gate(qk_chunk)).view(end - start, N, self.num_heads, 1))
            pred = torch.cat(pred_chunks, dim=0)
            eta = torch.cat(eta_chunks, dim=0)
        else:
            pred = self.learner(qk_state)
            eta = torch.sigmoid(self.lr_gate(qk_input)).view(BHW, N, self.num_heads, 1)

        corr = None
        conf = None
        context_residual = 0.0
        if self.target_mode == "epipolar" and stereo_context is not None:
            left_feat = stereo_context["left_feat"]
            right_feat = stereo_context["right_feat"]
            corr = stereo_context.get("corr", None)
            corr = self._ensure_corr_3d(corr, BHW, N, x.device, x.dtype)
            left_embed, right_embed, corr_embed = self.context_proj(
                left_feat, right_feat, corr
            )
            if self.epipolar_target_style == "split_ln" and self.ln_diff is not None:
                diff = right_embed - left_embed
                corr_gate = torch.sigmoid(self.corr_gate(torch.cat([diff, corr_embed], dim=-1)))
                target_full = self.ln_diff(diff) + corr_gate * self.ln_corr(corr_embed)
            else:
                target_full = right_embed - left_embed + corr_embed
                target_full = self.target_norm(target_full)
            context_residual = (left_embed + right_embed + corr_embed) * 0.0
        else:
            target_full = rearrange(v - k, "b n h d -> b n (h d)")
            target_full = self.target_norm(target_full)
            if stereo_context is not None:
                left_feat = stereo_context["left_feat"]
                right_feat = stereo_context["right_feat"]
                corr = stereo_context.get("corr", None)
                corr = self._ensure_corr_3d(corr, BHW, N, x.device, x.dtype)
                left_embed, right_embed, corr_embed = self.context_proj(
                    left_feat, right_feat, corr
                )
                context_residual = (left_embed + right_embed + corr_embed) * 0.0

        if self.stop_grad_target:
            target_full = target_full.detach()
        target_full = target_full + context_residual
        target = rearrange(target_full, "b n (h d) -> b n h d", h=self.num_heads)

        err = pred - target
        loss_token, update_dir = self._update_dir_and_loss_token(err)
        eta = eta * self.base_lr / float(self.head_dim)

        if corr is not None and (self.use_confidence_weight or self.router_loss_weight_conf):
            conf = self._confidence_from_corr(corr, BHW, N)

        if self.use_confidence_weight and conf is not None:
            # conf [BHW,N] -> [BHW,N,1,1] so it broadcasts with eta [BHW,N,H,1] (not [BHW,N,1], which mis-aligns N vs H).
            eta = eta * (0.1 + 0.9 * conf.unsqueeze(-1).unsqueeze(-1))

        if self.router_loss_weight_conf and conf is not None:
            w = (0.1 + 0.9 * conf).detach()
            loss_token = loss_token * w

        adapt = pred - eta * update_dir

        hard_mask = self.router(loss_token)
        adapt = adapt * hard_mask.float()
        adapt = rearrange(adapt, "b n h d -> b n (h d)")

        if self.spatial_fusion is not None:
            adapt = rearrange(adapt, "(b h w) n c -> (b n) c h w", b=B, h=H, w=W)
            adapt = self.spatial_fusion(adapt)
            adapt = rearrange(adapt, "(b n) c h w -> (b h w) n c", b=B, n=N)

        msg = self.o_proj(adapt)
        if self.use_residual:
            out = shortcut + self.drop_path(self.proj_drop(self.msg_gamma * msg))
        else:
            out = self.drop_path(self.proj_drop(self.msg_gamma * msg))

        if self.normalize_before:
            out = out + self.drop_path(self.mlp(self.norm2(out)))
        else:
            out = self.norm1(out)
            out = out + self.drop_path(self.mlp(out))
            out = self.norm2(out)
        return out

    def forward(self, label_rep, abs_encoding, attn_mask=None, stereo_context=None):
        return self._forward_impl(label_rep, abs_encoding, stereo_context=stereo_context)
