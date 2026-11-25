# Experimental protocol

## Data separation

The split seed is 2027. Within each class, shuffled source groups are allocated approximately 60% to training, 10% to checkpoint tuning, 10% to detector calibration, and 20% to testing. Rounding occurs at group level. Kather uses individual tiles as groups because no validated patient mapping is available in this reconstruction. SIPaKMeD uses a class-qualified source-cluster ID parsed from each cropped cell filename. This prevents cells from the same cluster appearing in different partitions; it does not establish patient separation.

Exact duplicates are detected from decoded original RGB pixels and dimensions before resizing. Source file hashes and pixel hashes are separate fields. The manifest is the definitive record of retained samples and partitions.

A separate Kather sensitivity analysis uses inferred filename families `CRC-Prim-HE-01` through `-10`. A permutation with seed 2027 assigns six complete families to training, one to tuning, one to calibration, and two to testing. Width 8 is retrained with seed 17; clean classification, PGD-8/255 classification, and detector metrics are reported separately. These inferred families are not independently verified patient IDs. This analysis does not change or relabel the primary tile-level partition.

## Architecture and training

Each compact CNN contains three blocks with channels `(w, 2w, 4w)`, for `w = 8, 16, 32`. A block has two bias-free 3 × 3 convolutions, group normalization with four groups, ReLU, and then 2 × 2 max pooling. Adaptive average pooling and a linear layer produce logits. Inputs are resized to 64 × 64 with Pillow bilinear interpolation and scaled to [0,1]. Only training images receive horizontal and vertical flips and quarter-turn rotations.

The proxy is parameter count divided by the mean global parameter-gradient norm on three fixed training batches. The selected candidate minimizes this ratio. Each initialization seed uses the same probe batches across candidates. Training begins from a fresh reset of the seed, so proxy evaluation does not affect the training RNG sequence. All candidates are trained to audit the small pool; this additional study cost is separate from selecting and training one candidate in use.

Training uses AdamW, initial learning rate 0.001, weight decay 0.0001, batch size 64, cosine decay, and 25 epochs. Seeds are 17, 29, and 41. The checkpoint with the best tuning macro F1 is retained; ties retain the earliest epoch. No test result selects a checkpoint or architecture.

## Detector calibration

At inference, the detached predicted class is used as the label for input-gradient cross-entropy. Losses are summed across a minibatch before differentiation, preserving each image's gradient scale. Group normalization makes outputs independent of other images in the batch.

The gradient scale is the maximum gradient norm on clean calibration images. It remains fixed for all test inputs. The normalized gradient is clipped to [0,1]; the combined score gives it weight 0.5 and `1 - maximum softmax probability` weight 0.5. Separate thresholds for combined score, normalized gradient, uncertainty, and entropy are the higher-order-statistic 95th percentiles on the same clean calibration partition. A score must be strictly greater than the threshold to trigger a flag.

## Stress tests and metrics

FGSM and random-start PGD use true labels. Pixel values remain in [0,1]. The L-infinity budgets are 4/255 and 8/255. PGD uses 20 iterations, a 2/255 step, and one uniform random start. Gaussian noise has standard deviation 8/255 and is clipped to the valid pixel range; it has no L-infinity budget.

The adaptive objective maximizes true-label cross-entropy minus the combined detector score, with coefficient 1. It uses second derivatives through the input gradient and the same PGD budget and step count. It runs on 128 sampled initially correct images per seed, using seed + 10000 for selection. It is a limited detector-aware stress test rather than an exhaustive attack.

The saved outputs distinguish these denominators:

| Metric | Population |
| --- | --- |
| Classification accuracy and macro F1 | Every evaluated clean or perturbed image |
| ASR | Initially correct images that become incorrect |
| TPR on all perturbed inputs | Every perturbed image |
| TPR on successful attacks | Initially correct images that become incorrect |
| FPR | Matching clean images; the adaptive test uses its clean subset |
| AUROC | Matching clean and perturbed images, with equal counts |

ECE uses 15 equal-width confidence bins. NLL clips the probability at 1e-12 only for the logarithm. The multiclass Brier score sums squared class-probability errors before averaging across images. Macro F1 includes every dataset class, including classes absent from a restricted subset. Three-seed standard deviations describe initialization variation on the fixed partition.

## Semantic and federated checks

The reference graph contains clean and PGD-8/255 predictions for seed 17. Every prediction has source/input/model identifiers and numeric signals. SHACL checks required links, cardinalities, domains, and stage-status consistency. Invalid confidence, missing links, and unsupported externally supplied stage status are negative controls. The SPARQL count of flagged prediction entities must equal the event count. This establishes schema consistency, not clinical truth.

The FedAvg experiment uses three balanced IID clients, five rounds, one local epoch per round, and seed 17. Client weights are sample counts. The final round is predeclared; test trajectories do not select a round. It is one local process without secure aggregation, differential privacy, real institutions, or communication-network measurements.

