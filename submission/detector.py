"""판별 모델 본체 — MoE(전문가 5) + Common(CNN), 원본 해상도 (docs/전략.md 12-2, 결정 ⑫).

이 파일은 '모델의 뼈대'만 담는다. 이미지 읽기·확률 반환은 inference.py, 학습은 별도 스크립트.

흐름 (사진 한 장 기준, 1024×768이면 조각 12장)
    원본 ─ 256×256 조각 12장 ─▶ DINOv3(백본 A) ─▶ 패치 토큰 12×256 = 3072개
                                    ├─ Gate      : 어텐션 풀링 → softmax(5)   "어느 계열 흔적 같은가"
                                    └─ Expert 1~5: 어텐션 풀링 → 특징+점수     "계열별 진짜/가짜 전문가"
    원본 한 장 그대로 ─▶ CNN(Common) ─────────────────────────────────────────┐
    [Gate로 섞은 Expert 특징·점수, Common 특징] 이어붙이기 ─▶ MLP ─▶ 로짓(시그모이드 전 값)

⚠️ 조각은 '자르기만' 한다. 리사이즈 금지 — 픽셀 수준 흔적은 축소하면 사라진다 (12-1).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file, save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel


def tile_image(x, tile):
    """[B,3,H,W] → [B*조각수,3,tile,tile]. 사진 순서가 앞이고 조각은 위→아래, 왼→오른 순. 겹침·리사이즈 없음."""
    _, c, h, w = x.shape
    if h % tile or w % tile:
        raise ValueError(f'이미지 크기 {w}×{h}가 조각 크기 {tile}의 배수가 아님')
    t = x.unfold(2, tile, tile).unfold(3, tile, tile)          # [B,C,gh,gw,tile,tile]
    return t.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, tile, tile)


class AttnPool(nn.Module):
    """어텐션 풀링: 토큰 수천 개 중 '중요한 위치'에 가중치를 줘서 벡터 하나로 모은다.
    학습되는 질문 벡터 1개가 토큰들을 훑는다. 평균과 달리 사진 일부에만 있는 흔적(부분 편집)이 묻히지 않는다."""

    def __init__(self, in_dim, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)                      # 1024차원 토큰을 작게 줄여 가볍게 한다
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, tokens):                                  # [B,N,in_dim] → [B,dim]
        kv = self.proj(self.norm(tokens))
        out, _ = self.attn(self.query.expand(tokens.shape[0], -1, -1), kv, kv, need_weights=False)
        return out[:, 0]


class Expert(nn.Module):
    """한 계열(E1~E5) 전문가: 풀링 → 작은 MLP → 특징 벡터, 그리고 자기 점수(1단계 학습에서 이 점수로 손실을 준다)."""

    def __init__(self, in_dim, pool_dim, heads, out_dim):
        super().__init__()
        self.pool = AttnPool(in_dim, pool_dim, heads)
        self.mlp = nn.Sequential(nn.Linear(pool_dim, pool_dim), nn.GELU(), nn.Linear(pool_dim, out_dim))
        self.score = nn.Linear(out_dim, 1)

    def forward(self, tokens):
        f = self.mlp(self.pool(tokens))
        return f, self.score(f).squeeze(-1)


class Gate(nn.Module):
    """입력이 어느 계열 흔적에 가까운지 → 전문가 5명에 대한 가중치(합 1)."""

    def __init__(self, in_dim, pool_dim, heads, n_experts):
        super().__init__()
        self.pool = AttnPool(in_dim, pool_dim, heads)
        self.mlp = nn.Sequential(nn.Linear(pool_dim, pool_dim), nn.GELU(), nn.Linear(pool_dim, n_experts))

    def forward(self, tokens):
        return self.mlp(self.pool(tokens)).softmax(-1)


class TopKPool(nn.Module):
    """Common CNN 마지막 풀링: 평균 대신 '채널마다 값이 큰 위치 상위 frac만 평균'(top-k pooling).
    작은 영역만 가짜인 사진(부분 편집)에서 그 영역의 반응이 전체 평균에 묻히지 않게 한다. mix=0이면 기존 평균과 같다.
    파라미터가 없어서 기존 가중치(state_dict)가 그대로 로드된다. 출력 크기 [B,C]는 평균 풀링과 같다."""

    def __init__(self, frac=0.1, mix=0.5):
        super().__init__()
        self.frac, self.mix = frac, mix

    def forward(self, x):
        flat = x.flatten(2)                                     # [B,C,H*W]
        avg = flat.mean(-1)
        k = max(1, int(round(self.frac * flat.shape[-1])))
        top = flat.topk(k, dim=-1).values.mean(-1)
        return (1 - self.mix) * avg + self.mix * top


class SpectrumHead(nn.Module):
    """사진 한 장의 '흑백 전역 스펙트럼 요약' → 작은 MLP → out_dim 벡터 (docs/전략.md 가이드 FFT 항목을 위치 비의존 숫자로 바꾼 것).
    흑백(휘도)으로 바꾼 뒤 Hann 창을 씌워 2D FFT → log10 파워에서 ① 반지름 방향 평균(32칸, 블러·노이즈 바닥) ② 각도별 에너지(12칸)
    ③ JPEG 블록 격자 흔적(경계 위치 차분이 경계가 아닌 위치보다 큰 정도, 가로·세로 × 8·16픽셀 주기 4개)을 뽑는다. 사진마다 min-max로 누르지 않는다(노이즈 양 정보가 사라지므로).
    FFT 지도는 주파수 좌표라 픽셀과 위치가 안 맞아 CNN 채널로 못 쓰고, 이렇게 숫자로 요약해서 최종 MLP에만 붙인다. 특징 계산에는 학습 파라미터가 없다."""
    N_RAD, N_ANG, N_BLOCK = 32, 12, 4
    N_FEAT = N_RAD + N_ANG + N_BLOCK

    def __init__(self, mean, std, out_dim=32):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer('px_mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('px_std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(self.N_FEAT, 64), nn.GELU(), nn.Linear(64, out_dim))
        self._maps = {}

    def _grid(self, h, w, dev):
        key = (h, w, str(dev))
        if key not in self._maps:
            fy = torch.fft.fftfreq(h, device=dev).view(-1, 1)                 # [H,1] cycles/pixel
            fx = torch.fft.rfftfreq(w, device=dev).view(1, -1)                # [1,W/2+1]
            r = torch.sqrt(fx ** 2 + fy ** 2)
            edges = torch.logspace(math.log10(1 / 32), math.log10(0.5), self.N_RAD + 1, device=dev)
            rad = torch.bucketize(r, edges[1:-1])                             # 0..N_RAD-1
            rad_ok = (r >= edges[0]) & (r <= edges[-1])
            ang = (torch.atan2(fy, fx.expand_as(r)) % math.pi) / math.pi * self.N_ANG
            ang = ang.long().clamp(max=self.N_ANG - 1)
            ang_ok = (r >= 0.08) & (r <= 0.5)
            win = torch.hann_window(h, periodic=False, device=dev).view(-1, 1) * torch.hann_window(w, periodic=False, device=dev).view(1, -1)
            self._maps[key] = (rad.flatten(), rad_ok.flatten(), ang.flatten(), ang_ok.flatten(), win)
        return self._maps[key]

    @staticmethod
    def _binmean(p, idx, ok, n):
        """p [B,K] 파워를 idx(0..n-1)별 평균 → [B,n]. ok가 False인 자리는 뺀다."""
        b = p.shape[0]
        s = torch.zeros(b, n, device=p.device, dtype=p.dtype).index_add_(1, idx[ok], p[:, ok])
        c = torch.zeros(n, device=p.device, dtype=p.dtype).index_add_(0, idx[ok], torch.ones(int(ok.sum()), device=p.device, dtype=p.dtype))
        return s / c.clamp(min=1)

    @torch.no_grad()
    def features(self, x):
        """x [B,3,H,W] 정규화된 RGB → [B, N_FEAT] (고정 상수로 대략 O(1)이 되게 맞춤). H,W는 8의 배수여야 한다(대회 규격은 만족)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            b, _, h, w = x.shape
            rgb = x.float() * self.px_std + self.px_mean
            g = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
            g = g - g.mean(dim=(1, 2), keepdim=True)
            rad, rad_ok, ang, ang_ok, win = self._grid(h, w, x.device)
            f = torch.fft.rfft2(g * win)
            p = (f.real ** 2 + f.imag ** 2) / win.pow(2).sum()               # [B,H,W/2+1] 파워
            pf = p.reshape(b, -1)
            radial = torch.log10(self._binmean(pf, rad, rad_ok, self.N_RAD) + 1e-12)
            angular = torch.log10(self._binmean(pf, ang, ang_ok, self.N_ANG) + 1e-12)
            # JPEG 블록 격자: 인접 픽셀 차분의 크기를 다른 축으로 평균해 1차원 프로필을 만들고, 블록 경계 위치(8·16의 배수)의 값 ÷ 블록 안쪽 위치의 값.
            # (부호가 제멋대로인 경계 불연속은 1차 스펙트럼에 봉우리를 만들지 않으므로 크기(절댓값)로 본다)
            gx = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(1)                  # [B,W-1], gx[i] = |I(i+1) - I(i)|
            gy = (g[:, 1:, :] - g[:, :-1, :]).abs().mean(2)                  # [B,H-1]
            blocks = []
            for gv in (gx, gy):
                pos = torch.arange(gv.shape[1], device=gv.device) + 1
                for period in (8, 16):
                    on = gv[:, pos % period == 0].mean(1)
                    ref = gv[:, pos % period == period // 2].mean(1)
                    blocks.append(torch.log10(on + 1e-9) - torch.log10(ref + 1e-9))
            peaks = torch.stack(blocks, 1)                                   # [B,4]
            return torch.cat([(radial + 4.0) / 2.0, (angular + 5.0) / 1.5, peaks / 0.15], 1)

    def forward(self, x):
        return self.mlp(self.features(x))


def with_highpass(x, gain):
    """Common CNN 입력에 '고주파 잔차'를 3채널 더 붙인다: 정규화된 RGB x [B,3,H,W] → [B,6,H,W].
    잔차 = x − (3×3 평균) — 블러·노이즈·인페인팅 경계처럼 미세한 픽셀 흔적을 gain배로 키워 CNN이 쉽게 보게 한다.
    공간 위치가 RGB와 그대로 대응한다(FFT 크기 지도는 주파수 좌표라 CNN 입력으로 부적절해서 쓰지 않음). 파라미터 없음."""
    r = x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, 1)
    return torch.cat([x, r * gain], 1)


class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tile = cfg['tile']
        n = cfg['n_experts']
        # 구조는 config로 만든다 — from_pretrained·허브 이름 호출 없음(오프라인, 12-8 #1). 가중치는 load_weights로 따로 읽는다.
        self.backbone = DINOv3ViTModel(DINOv3ViTConfig.from_dict(cfg['backbone']))
        d = self.backbone.config.hidden_size
        self.n_prefix = 1 + self.backbone.config.num_register_tokens   # 맨 앞 CLS + 레지스터 토큰은 패치가 아니므로 뺀다
        self.gate = Gate(d, cfg['pool_dim'], cfg['pool_heads'], n)
        self.experts = nn.ModuleList(Expert(d, cfg['pool_dim'], cfg['pool_heads'], cfg['expert_dim']) for _ in range(n))
        hp = cfg['common'].get('highpass')                       # 기본(키 없음)은 RGB 3채널 — 기존 제출물과 동작 동일
        self.common = timm.create_model(cfg['common']['name'], pretrained=False, num_classes=0, in_chans=6 if hp else 3)   # 풀링까지만 → 특징 벡터
        if hp:                                                   # 입력을 forward 직전에 바꾸는 훅 — model.common(x) 호출부(학습·진단·추론)는 그대로 둔다
            gain = float(hp['gain'])
            self.common.register_forward_pre_hook(lambda m, args: (with_highpass(args[0], gain),) + tuple(args[1:]))
        if cfg['common'].get('pool') == 'topk':                  # 기본(키 없음)은 평균 풀링 — 기존 제출물과 동작 동일
            self.common.global_pool = TopKPool(cfg['common'].get('frac', 0.1), cfg['common'].get('mix', 0.5))
        sp = cfg.get('spectrum')                                 # 기본(키 없음)은 없음 — 기존 제출물과 동작 동일
        if sp:                                                   # 흑백 전역 스펙트럼 요약을 Common 출력 뒤에 붙인다 → model.common(x)가 [CNN 특징 | 스펙트럼 벡터]를 낸다(호출부 그대로)
            self.spec = SpectrumHead(cfg['mean'], cfg['std'], int(sp.get('dim', 32)))
            self.common.num_features = self.common.num_features + self.spec.out_dim
            self.common.register_forward_hook(lambda m, args, out: torch.cat([out, self.spec(args[0][:, :3])], 1))   # args[0]은 (highpass가 있으면 6채널) 입력 — 앞 3채널이 RGB
        self.head = nn.Sequential(
            nn.Linear(cfg['expert_dim'] + 1 + self.common.num_features, cfg['head_hidden']),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(cfg['head_hidden'], 1))
        # config의 fusion_norm이 true이면 합치기 전에 Expert 벡터(64)와 Common 벡터(1280)에 각각 LayerNorm을 건다(exp08).
        # 이유: head 입력 표준편차가 Expert 10.6·Gate 점수 30.9·Common 0.22로 50배 넘게 달랐다(10/5 diag_usage).
        # 꺼져 있으면(기본) 모듈이 없어 기존 가중치(exp02 등)가 그대로 로드된다.
        self.fusion_norm = nn.ModuleList([nn.LayerNorm(cfg['expert_dim']), nn.LayerNorm(self.common.num_features)]) if cfg.get('fusion_norm') else nn.ModuleList()
        self.backbone_trainable = False     # 1단계는 백본 고정. 4단계(LoRA 등)에서 True
        self.mod_drop = 0.0                 # 학습 때만 쓰는 값(train.py --mod_drop이 3단계에서 켬). 저장·추론에는 영향 없음

    def get_tokens(self, images):
        """images: [B,3,H,W] → 백본 패치 토큰 [B, 조각수×256, D] (fp32). 학습 스크립트도 이 함수를 쓴다.
        config의 tile이 0이면 조각 없이 사진 전체를 한 번에 백본에 넣는다(exp07 실험): 토큰 수는 같지만(1024×768 → 3072) 전역 어텐션으로
        조각 경계를 넘는 문맥이 보인다. 리사이즈는 하지 않고(해상도 그대로), 계산량은 약 2배다(어텐션이 N²)."""
        b = images.shape[0]
        if self.tile:
            tiles = tile_image(images, self.tile)
        else:
            p = self.backbone.config.patch_size
            if images.shape[-2] % p or images.shape[-1] % p:
                raise ValueError(f'이미지 크기 {images.shape[-1]}×{images.shape[-2]}가 패치 크기 {p}의 배수가 아님')
            tiles = images
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.backbone_trainable):
            hs = self.backbone(pixel_values=tiles.to(next(self.backbone.parameters()).dtype)).last_hidden_state
        tokens = hs[:, self.n_prefix:, :].float()               # 패치 토큰만 [B*조각수, 256, D]
        return tokens.reshape(b, -1, tokens.shape[-1])          # 사진 한 장의 토큰을 한 줄로 [B, 3072, D]

    def combine(self, tokens, common):
        """토큰 + Common(CNN) 특징 → Gate·Expert·최종 로짓. 학습 스크립트가 단계별로 따로 부를 수 있게 분리했다."""
        gate = self.gate(tokens)                                # [B,5]
        outs = [e(tokens) for e in self.experts]
        feats = torch.stack([o[0] for o in outs], 1)            # [B,5,expert_dim]
        scores = torch.stack([o[1] for o in outs], 1)           # [B,5]
        mixed = (gate.unsqueeze(-1) * feats).sum(1)             # Weighted Sum
        mixed_score = (gate * scores).sum(1, keepdim=True)
        if len(self.fusion_norm):                               # Gate 점수(스칼라 1개)는 LayerNorm이 의미가 없어 그대로 둔다
            mixed, common = self.fusion_norm[0](mixed), self.fusion_norm[1](common)
        feat = torch.cat([mixed, mixed_score, common], 1)       # 최종 MLP 입력 (SupCon 대조 학습도 이걸 쓴다 — 드롭아웃 전 값)
        x = self._modality_dropout(feat) if (self.training and self.mod_drop > 0) else feat
        return {'logit': self.head(x).squeeze(1), 'gate': gate, 'expert_logits': scores, 'feat': feat}

    def _modality_dropout(self, feat):
        """Feature(modality) dropout: 사진마다 확률 p로 Expert 블록(앞쪽 expert_dim+1개)을, 다른 확률 p로 Common 블록(나머지)을 0으로 만든다.
        한 사진에서 둘이 같이 꺼지지는 않는다. 최종 MLP가 한쪽 입력(특히 Common)에만 기대지 않게 하려는 학습용 장치 — 추론에는 쓰이지 않는다."""
        n_e = feat.shape[1] - self.common.num_features
        r = torch.rand(feat.shape[0], device=feat.device)
        keep_e = (r >= self.mod_drop).to(feat.dtype)                                  # r < p → Expert 끔
        keep_c = ((r < self.mod_drop) | (r >= 2 * self.mod_drop)).to(feat.dtype)      # p ≤ r < 2p → Common 끔
        return torch.cat([feat[:, :n_e] * keep_e[:, None], feat[:, n_e:] * keep_c[:, None]], 1)

    def forward(self, images):
        """images: [B,3,H,W] 정규화된 실수 텐서 (H,W는 조각 크기의 배수). 사진마다 독립 계산 — 사진끼리 섞이지 않는다."""
        return self.combine(self.get_tokens(images), self.common(images))   # CNN은 원본 한 장 그대로 (크기 제한 없음)

    def prepare(self, device):
        """추론용 준비: 가중치는 항상 fp32. ⚠️ 백본을 .half()로 바꾸면 DINOv3 ViT-L이 T4에서 첫 블록부터 inf/NaN이 난다(10/2 실측).
        GPU 속도는 inference.py가 autocast(fp16)로 얻는다 — 연산만 fp16이고 잔차는 fp32라 안전하다."""
        self.to(device)
        return self.eval()


def save_weights(model, path):
    """백본은 fp16(약 0.6GB), 나머지는 fp32로 safetensors 한 파일에 저장 → 1GB 한도 안."""
    state = {k: (v.detach().cpu().half() if k.startswith('backbone.') else v.detach().cpu().float()).contiguous()
             for k, v in model.state_dict().items()}
    save_file(state, str(path))


def load_weights(model, path):
    model.load_state_dict(load_file(str(path)), strict=True)    # 키가 하나라도 안 맞으면 바로 에러


# ── 학습 시작용: 공식 사전학습 가중치 파일을 우리 모델에 넣는다 (제출 추론에서는 쓰지 않음 — 제출은 위 load_weights)
def load_pretrained_backbone(model, path):
    """공식 DINOv3 파일(키 'layer.N...')을 백본에 넣는다. transformers 5.16.1의 모델은 'model.layer.N...'이라
    모델에 없는 이름이고 'model.'+이름이 있을 때만 바꿔 준다(나머지 7개 키는 그대로). 키가 안 맞으면 에러."""
    sd = load_file(str(path))
    keys = set(model.backbone.state_dict().keys())
    sd = {(('model.' + k) if (k not in keys and ('model.' + k) in keys) else k): v for k, v in sd.items()}
    model.backbone.load_state_dict(sd, strict=True)


def load_pretrained_common(model, path):
    """timm efficientnet_b0 사전학습 파일을 Common CNN에 넣는다.
    입력이 6채널(common.highpass)이면 첫 합성곱의 새 채널 가중치를 0으로 채운다 → 시작 시점엔 사전학습 RGB 모델과 똑같이 동작하고, 학습하면서 잔차를 쓰기 시작한다."""
    sd = load_file(str(path))
    w = sd['conv_stem.weight']
    n_in = model.common.conv_stem.weight.shape[1]
    if n_in != w.shape[1]:
        sd['conv_stem.weight'] = torch.cat([w, torch.zeros(w.shape[0], n_in - w.shape[1], *w.shape[2:], dtype=w.dtype)], 1)
    model.common.load_state_dict(sd, strict=True)
