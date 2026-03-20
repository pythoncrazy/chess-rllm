# Checkpoint Registry

All SFT runs use `Qwen/Qwen3.5-35B-A3B` as the base model with LoRA rank 32.
Local records are in `checkpoints/sft/<run>/checkpoints.jsonl`.
Tinker auto-expires checkpoints after ~2 days — only the most recent weights survive.

---

## SFT Runs

### `top3-cot-bs512` — 2026-03-15
- **Data**: SF vs SF games, top-3 CoT format (UCI notation)
- **Checkpoints**: 8 (final saved)
- **Tinker**: `tinker://3d80bfea-3d6c-5410-9d31-4fe50f046ca4:train:0/weights/final`
- **Status**: Completed — expired on Tinker

---

### `sf-vs-sf-v1` — 2026-03-16
- **Data**: SF vs SF games, standard format (UCI notation)
- **Checkpoints**: 8 (final saved)
- **Tinker**: `tinker://f4518001-8953-5924-a0a5-60a6a9859939:train:0/weights/final`
- **Status**: Completed — expired on Tinker

---

### `uci-v3` — 2026-03-17
- **Data**: Opening-only FENs (UCI notation)
- **Checkpoints**: 5 (stopped at step 250)
- **Tinker**: `tinker://71a04e26-534d-5713-b2ed-b85217e64b53:train:0/weights/000250`
- **Status**: Aborted early — expired on Tinker

---

### `uci-v4` — 2026-03-17
- **Data**: Opening-only FENs (UCI notation)
- **Checkpoints**: 1 (stopped at step 50)
- **Tinker**: `tinker://43f4ee6f-1ea2-5720-a782-3cbc4df6e986:train:0/weights/000050`
- **Status**: Aborted very early — expired on Tinker

---

### `sf-vs-sf-uci-v1` — 2026-03-17
- **Data**: SF vs SF full games, UCI notation
- **Checkpoints**: 20 (final saved)
- **Tinker**: `tinker://87ae2bb5-e889-52dd-a587-935e9a3e696a:train:0/weights/final`
- **Status**: Completed — expired on Tinker

---

### `uci-v5` — 2026-03-17
- **Data**: Mixed SF games + openings, UCI notation
- **Checkpoints**: 20 (final saved)
- **Tinker**: `tinker://eb8eb9a1-18fe-54fa-bebf-98f9335be0b7:train:0/weights/final`
- **Status**: Completed — expired on Tinker

---

### `uci-v6` — 2026-03-17
- **Data**: Mixed SF games, UCI notation
- **Checkpoints**: 5 (stopped at step 250)
- **Tinker**: `tinker://e9fc887b-25e5-5780-a697-c092d98670dd:train:0/weights/000250`
- **Status**: Aborted early — expired on Tinker

---

### `uci-v7` — 2026-03-18
- **Data**: 10M SF games (noob_5moves + UHO_4060_v4 openings), UCI notation
- **Checkpoints**: 58 (final saved), save_freq=50
- **Tinker**: `tinker://c5e88bc9-cede-5a39-a8c7-97347447111f:train:0/weights/final`
- **Status**: Completed — expired on Tinker

---

### `uci-v8` — 2026-03-18 ⚠️ last surviving local record
- **Data**: 10M SF games (noob_5moves + UHO_4060_v4 openings), SAN notation switch mid-run
- **Checkpoints**: 199 local entries, last at step 9950 (~half epoch of 10M dataset)
- **Tinker**: `tinker://6e7aac48-4d40-538f-92a9-7d67e5ebbef9:train:0/weights/009950`
- **Status**: Run marked corrupted on Tinker (crashed/billing issue); weights checkpoint still live (6.7 GB)
- **Note**: This is the **only checkpoint still accessible on Tinker**

---

## GRPO Runs

GRPO ran from the `uci-v8` SFT checkpoint. All GRPO checkpoints have expired on Tinker.

### `uci-grpo-v9` — 2026-03-18
- **Base**: `uci-v8` final SFT weights
- **Data**: `grpo_openings.jsonl` (UHO_4060_v4 full games, SAN notation)
- **Config**: group_size=8, batch=32 positions × 8 rollouts = 256 rollouts/step
- **Reward**: Stockfish centipawn-loss `exp(-loss/50)`
- **Status**: Ran to step ~9950, all checkpoints expired on Tinker

---

## Upcoming

### `phase1-chess-500k` (planned)
- **Base**: `Qwen/Qwen3.5-35B-A3B` (fresh)
- **Data**: First 500k of `sf_train_10m.jsonl` (SAN notation)
- **Config**: batch=512, lr=1e-3, 1 epoch
- **Local dir**: `checkpoints/sft/phase1-chess-500k/`

### `phase2-english` (planned)
- **Base**: Resumes from `phase1-chess-500k` final checkpoint
- **Data**: `english_50k.jsonl` (18,756 Gemini-generated English explanations)
- **Config**: batch=32, lr=1e-4, 2 epochs
- **Local dir**: `checkpoints/sft/phase1-chess-500k/` (same dir, auto-resume)
