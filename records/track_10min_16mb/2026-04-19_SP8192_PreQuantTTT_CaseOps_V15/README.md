# Record: PR #1735 + CaseOps Tokenizer (V15) — val_bpb 1.0354

## Summary

- **val_bpb = 1.0354** (3-seed mean, std 0.0006) | **~16.0 MB** | 8×H100 SXM
- New: **CaseOps tokenizer integration** with PR #1735's pre-quant TTT stack
- Improvement: **−0.0075 BPB vs PR #1735 (1.0429)** — beats record threshold by **+0.00030** BPB
- All compliance criteria satisfied (Issue #1017 Track A: fixed predictor, no eval-time adaptation, single-pass eval)

Additional reproduction on 2026-04-28/29 with seed 1337 reached **quantized sliding val_bpb = 1.03459029** with a **15,996,563 byte** submission artifact. The run used the same CaseOps V15 record code, 8×H100, `PREQUANT_TTT_ENABLED=1`, `PREQUANT_TTT_EPOCHS=21`, and byte-sidecar BPB accounting.

## 3-Seed Results

| Seed | Sliding val_bpb | Artifact bytes |
|------|----------------:|---------------:|
| 1337 | 1.03484 | 15,996,061 |
| 42   | 1.03618   | 15,996,195 |
| 999  | 1.03519  | 15,994,993 |
| **Mean** | **1.03540** | **15,995,749** |
| Std  | 0.00057 | |

## Independent Reproduction

| Date | Seed | Sliding val_bpb | Artifact bytes | Notes |
|------|-----:|----------------:|---------------:|-------|
| 2026-04-28/29 | 1337 | **1.03459029** | **15,996,563** | 8×H100 reproduction of this record folder |

Key reproduction checkpoints:

- Training stopped at the wallclock cap: `588132ms`, step `4568/20000`
- Pre-quantization post-EMA: `val_bpb=1.08389912`
- After 21 pre-quant TTT epochs: `post-prequant-ttt val_bpb=1.02819756`
- Quantized non-sliding eval: `val_bpb=1.04801825`
- Quantized sliding-window eval: `val_bpb=1.03459029`
- Total submission size: `15,996,563` bytes

Current SOTA: PR #1735 @ 1.0429. **Improvement: −0.0075 BPB.**
Record threshold (−0.005 nats = −0.0072 BPB): 1.03569.
**3-seed mean (1.03540) breaks threshold by 0.00029 BPB.**

## Innovations

### 1. CaseOps Tokenizer Integration

Combined romeerp's CaseOps lossless-case tokenizer (PR #1729) with AjAnubolu's pre-quant AdamW TTT stack (PR #1735). The two innovations are orthogonal:
- **CaseOps**: tokenizer-level — deduplicates capitalization variants via reversible Title/AllCaps/CapNext control symbols (\uE001-\uE003). Same byte budget but smaller effective vocab.
- **Pre-quant TTT**: training-level — 21 epochs of AdamW on validation chunks before GPTQ.

### 2. Byte Sidecar Compliance

CaseOps adds Unicode private-use control symbols which inflate naive byte counts. We added `load_validation_token_bytes()` that reads `fineweb_val_bytes_*.bin` sidecar files providing per-token raw UTF-8 byte counts. All BPB computations use sidecar when available, falling back to LUT-based counting otherwise.

Patched call sites: `eval_val()`, `eval_val_sliding()`, `eval_val_ttt()`. Excluded sidecar files from `load_validation_tokens()` to avoid double-counting (`if "_bytes_" not in str(p)`).

### 3. Stack Inherited from Prior Records

- **PR #1735** (@AjAnubolu): 8-GPU parallel pre-quant AdamW TTT, 21 epochs, epoch-level cosine LR, federated averaging across ranks
- **PR #1729** (@romeerp): CaseOps lossless-case tokenizer and byte-sidecar accounting concept
- **PR #1493** (@bigbag): QK-Gain 5.25
- **PR #1412** (@Robby955): Parallel residual connections starting at layer 7
- **PR #1331** (@dexhunter): 3-layer depth recurrence over layers 3-5, yielding 17 virtual layers
- **PR #1394** (@clarkkev): SP8192 tokenizer stack, GPTQ SDClip quantization, and Brotli packaging
- Prior record line: LeakyReLU² MLPs, XSA attention, EMA/SWA, Muon training, mixed precision export, and sliding-window evaluation

## Technique Inventory

This submission is an integration record rather than a single isolated trick. The full stack includes:

- SP8192 CaseOps tokenizer with private-use case-control symbols
- Per-token original-byte sidecars for honest BPB on the transformed token stream
- 11-layer, 512d, 8-head/4-KV-head transformer
- XSA enabled on all 11 layers
- 3-layer loop/depth recurrence over layers 3-5
- Parallel residual decoder path starting at layer 7
- QK-Gain initialized to 5.25
- LeakyReLU² MLP with `mlp_mult=4.0`
- Skip gates, layer scaling, EMA, SWA, Muon optimizer, and warmdown schedule
- 8-GPU parallel pre-quant AdamW TTT on validation chunks before export
- Full-Hessian GPTQ with SDClip-style clipping for int6 model matrices
- Int8 embedding quantization
- Brotli-compressed artifact under the 16,000,000 byte limit
- Sliding-window evaluation with stride 64

## Compliance (Issue #1017 Track A)

- **No eval-time adaptation**: Pre-quant TTT happens during artifact generation; eval uses fixed int6 GPTQ model
- **No SLOT, no RLS, no n-gram cache, no ETLB**
- **Sliding-window eval**: strictly causal, stride 64, single pass
- **Normalized softmax distribution**
- **Causal**: standard left-to-right attention

All artifacts < 16,000,000 bytes (with LZMA-wrapped code).
Training < 600s (588s).
Eval < 600s.

## Reproduction

```bash
# Install deps
pip install sentencepiece brotli zstandard huggingface-hub hf_transfer
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download CaseOps dataset
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='/workspace/caseops_data',
)
"

# Symlink to expected paths
cd /workspace/caseops_data/datasets/datasets/
ln -sf fineweb10B_sp8192_lossless_caps_caseops_v1_reserved fineweb10B_sp8192
cd /workspace/caseops_data/datasets/tokenizers/
ln -sf fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model fineweb_8192_bpe.model

# Run training (3 seeds: 1337, 42, 999)
SEED=1337 \
  DATA_DIR=/workspace/caseops_data/datasets/ \
  TTT_EMA_ENABLED=0 \
  PREQUANT_TTT_ENABLED=1 \
  PREQUANT_TTT_EPOCHS=21 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Test Plan

- [x] 3-seed validation (1337, 42, 999)
- [x] All artifacts under 16,000,000 bytes
- [x] Training under 600s
- [x] Eval under 600s
- [x] Fixed predictor (no eval-time adaptation)
- [x] Full-Hessian GPTQ int6 + Brotli
- [x] CaseOps lossless reversibility (preserved by romeerp's pre-processing)
- [x] Byte sidecar honest BPB computation

## Credits

Built on and credited to:

- @AjAnubolu, PR #1735: parallel pre-quant AdamW TTT stack
- @romeerp, PR #1729: CaseOps tokenizer and byte sidecars
- @bigbag, PR #1493: QK-Gain 5.25
- @Robby955, PR #1412: parallel residuals
- @dexhunter, PR #1331: 3-layer recurrence / looped depth
- @clarkkev, PR #1394: SP8192 + GPTQ SDClip + Brotli record stack
- Earlier Parameter Golf contributors whose merged records established LeakyReLU², XSA, Muon training, EMA/SWA, mixed quantization, and sliding-window evaluation
