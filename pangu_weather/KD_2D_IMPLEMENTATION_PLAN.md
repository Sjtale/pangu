# PanguLite2DAttentionPosEmbed KD Implementation Plan

## Fixed contract

- Student: the DeifiliaTo/PanguWeather `PanguLite2DAttentionPosEmbed` design,
  with 2D shifted-window attention without Earth-specific bias, learned
  `absolute_pos_embed` of shape `[1, 91, 288]`, spatial patch size `[8, 8]`,
  width 288, heads `[6, 12, 12, 6]`, depths `[2, 6, 6, 2]`, and a 69-channel
  output (`4` surface plus `5 x 13` upper-air channels).
- Teacher: the official 3D width-192 Pangu checkpoint. It remains frozen and
  is used only to initialize transferable tensors and produce KD targets.
- Runtime: one DCU, batch size one, automatic mixed precision where supported,
  gradient accumulation, atomic resumable checkpoints, and 100 epochs.

## Hybrid 3D-to-2D initialization

The initializer will start from the student's normal random initialization and
overwrite only transferable regions. This is important because the requested
student width (288) is larger than the official teacher width (192), so a full
shape-equal copy is impossible.

1. Patch embedding: merge the teacher's surface Conv2d and upper-air Conv3d
   kernels into the student's 72-input-channel Conv2d. Surface channels map
   directly. Each upper-air `(variable, level)` channel receives the matching
   teacher variable/depth slice. Spatial 4x4 kernels are bicubically resized to
   8x8 with fan-in compensation. Only the overlapping 192 output features are
   overwritten; the remaining 96 retain Kaiming initialization.
2. Patch recovery: merge the teacher's surface ConvTranspose2d and upper-air
   ConvTranspose3d filters into the student's 69-output ConvTranspose2d,
   mapping the 65 upper-air outputs by `(variable, level)` and resizing 4x4 to
   8x8. The overlapping 384 of 576 input features are copied.
3. MLPs and normalization: for every corresponding block, copy the maximal
   leading hyper-rectangle of `mlp` weights/biases and normalization vectors.
   This preserves teacher features while leaving the extra width randomized.
4. Downsample and upsample: copy maximal overlapping regions of all Linear and
   LayerNorm tensors. No tensor is silently resized across incompatible ranks.
5. Attention and position: explicitly exclude all 3D attention QKV/projection,
   Earth-bias tables/indices, shifted-window masks, and
   `absolute_pos_embed`. The 2D attention parameters retain Xavier/truncated
   normal initialization, and the position embedding retains truncated normal
   initialization (`std=0.02`).
6. Audit: emit copied-element counts, random-only tensor names, source/target
   checkpoint hashes, and fail unless every requested component category was
   transferred and every attention/position tensor remained untouched.

## Distillation objective

The 15 official channels are selected by the existing Xiandao index contract.
For prediction `p`, truth `y`, teacher `t`, latitude weights `w=cos(latitude)`,
and official-baseline RMSE values `b`:

`L = 0.55 * mean(RMSE_w(p15,y15)/b) + 0.30 * mean(1-ACC_w(p15,y15)) + 0.10 * MSE(p15,t15) + 0.05 * MSE(p54,t54)`.

ACC is computed on latitude-weighted spatial anomalies (weighted spatial mean
removed per sample/channel), matching the anomaly-correlation intent rather
than raw cosine similarity. Baseline RMSE values are mandatory and must contain
exactly 15 finite positive entries after normalization. The weak 54-channel
teacher term preserves the full output contract without competing strongly
with scored-channel optimization.

## Schedule and launch

- Epochs: 100, 2048 optimization steps per epoch unless explicitly reduced for
  a smoke test.
- Warm-up: the first 2 epochs use linear LR warm-up. During epoch 1, only 2D
  attention, absolute position, and recovery are trainable; epoch 2 unfreezes
  the whole student. A single cosine decay then continues through epoch 100.
- Effective batch: physical batch 1 with gradient accumulation 4; loss is
  divided by 4 and optimizer/scheduler steps occur only at accumulation
  boundaries.
- Checkpoint every 256 optimizer steps and at each epoch; resume must validate
  the architecture, loss weights, accumulation factor, and 100-epoch protocol.
- Preflight gates: strict teacher load, hybrid-transfer audit, one CPU shape
  smoke test on a reduced grid, one real-grid DCU forward/backward step, finite
  loss/gradients, and successful checkpoint resume. The long run starts only
  after these gates and only where the official checkpoint, baseline RMSE, ERA5
  mount, and DCU are present.

## Acceptance checks

- Student contains no Earth-specific bias parameter and has
  `absolute_pos_embed.shape == (1, 91, 288)`.
- Full-resolution output is surface `[1,4,721,1440]` plus upper air
  `[1,65,721,1440]` (or equivalent `[1,5,13,721,1440]` before flattening).
- Hybrid audit proves attention/position stayed random and all five requested
  transfer categories were populated.
- Loss unit tests verify latitude weighting, anomaly centering, baseline
  normalization, exact 15/54 split, and teacher-gradient detachment.
- Launcher proves `epochs=100`, `batch_size=1`, accumulation, warm-up staging,
  atomic resume, and single-device execution.
