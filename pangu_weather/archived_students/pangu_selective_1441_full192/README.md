# pangu_selective_1441_full192

Status: **REJECTED / ARCHIVED**  
Archived: 2026-07-13

## Structure

```yaml
patch_size: [2, 8, 8]
embed_dim: 96
num_heads: [3, 6, 6, 3]
depth_blocks: [1, 4, 4, 1]
window_size: [2, 6, 12]
mlp_ratio: 4
```

The student used deterministic official-full192 initialization with fixed
block mapping `[0]/[0,1,4,5]/[0,1,4,5]/[0]`, 286/286 trainable state-key
coverage, and zero randomly initialized trainable parameters. Recovery was
teacher-free and used all 69 organizer-weighted output channels.

## Rejection evidence

- Server weighted validation loss: `0.2003`.
- Epoch-1 training cumulative loss was approximately `0.2025`; further
  low-learning-rate recovery was not competitive.
- The deterministic provenance requirement was satisfied, but the migrated
  function was not preserved. Changing patch4 to patch8 collapses four spatial
  tokens into one; bicubic kernel migration cannot retain their separate
  attention behavior. Simultaneous width192-to-width96 reduction compounded
  the information loss.

This candidate must not be trained, ranked, packed, or submitted again. Its
code remains only for reproducibility. Both the launcher and direct training
reject it unless `PANGU_ALLOW_REJECTED_SELECTIVE_1441=1` is explicitly set.

## Server artifact archive

After stopping `distill_train.py`, run from `pangu_weather/`:

```bash
bash scripts/run_selective_1441_full192.sh archive
```

The command verifies each copied artifact by SHA256, writes
`data/checkpoints/archive/pangu_selective_1441_full192_rejected_20260713/manifest.json`,
then removes the corresponding active checkpoint/log files. Official full192
and accepted baseline checkpoints are never included.
