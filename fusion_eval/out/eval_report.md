# Multimodal evidence fusion — evaluation report

## Overall metrics
| metric | value |
|---|---|
| n | 6.0 |
| top1_accuracy | 0.6667 |
| top1_accuracy_covered | 0.75 |
| top3_accuracy_covered | 0.75 |
| precision_accepted | 0.5 |
| recall_positive | 0.2 |
| f1 | 0.2857 |
| false_positive_rate | 0.0 |
| abstention_rate | 0.3333 |
| ambiguous_rate | 0.3333 |
| ece | 0.2694 |
| brier | 0.0776 |
| latency_layer1_mean_s | 3706.45 |
| gpu_mem_mean_mb | 0.0 |
| weight_source_counts | {'quality_proportional_fallback (1/1 samples)': 6} |

## By difficulty
| difficulty | top1 | top1_covered | f1 | ece | abstention |
|---|---|---|---|---|---|
| blur | 1.0 | 1.0 | 0.0 | 0.1607 | 0.0 |
| conflict | 0.0 | 0.0 | 0.0 | 0.3243 | 0.0 |
| easy | 0.5 | 1.0 | 0.6667 | 0.34 | 0.5 |
| negative | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| small | 1.0 | 1.0 | 0.0 | 0.2526 | 0.0 |

## By modality availability
| modalities | n | top1 | top1_covered | f1 | abstention |
|---|---|---|---|---|---|
| logo+ocr+product | 5 | 0.6 | 0.75 | 0.2857 | 0.2 |
| none | 1 | 1.0 | 0.0 | 0.0 | 1.0 |

## Ablations (delta vs ALL)
| combo | top1 | f1 | ece | abstention | Δtop1 | Δf1 | Δece |
|---|---|---|---|---|---|---|---|
| ALL | 0.6667 | 0.2857 | 0.2694 | 0.3333 | 0.0 | 0.0 | 0.0 |
| NO_LOGO | 0.6667 | 0.0 | 0.165 | 0.5 | 0.0 | -0.2857 | -0.1044 |
| NO_OCR | 0.6667 | 0.3333 | 0.2709 | 0.3333 | 0.0 | 0.0476 | 0.0015 |
| NO_ASR | 0.6667 | 0.2857 | 0.2694 | 0.3333 | 0.0 | 0.0 | 0.0 |
| NO_PRODUCT | 0.1667 | 0.0 | 0.3243 | 0.8333 | -0.5 | -0.2857 | 0.0549 |
| VISUAL_ONLY | 0.6667 | 0.2857 | 0.2694 | 0.3333 | 0.0 | 0.0 | 0.0 |
| TEXT_ONLY | 0.1667 | 0.0 | 0.0 | 1.0 | -0.5 | -0.2857 | -0.2694 |

## Calibration
best temperature: 1.5  (fit ECE 0.2747, fit Brier 0.0978, n_fit 3)
held-out ECE 0.1764  held-out Brier 0.0311  n_test 1

## Per-video verdicts
| video | difficulty | gt | winner | prob | verdict | taxonomy |
|---|---|---|---|---|---|---|
| easy_ADIDAS | easy | ADIDAS | None | 0.0 | abstain | FN_UNKNOWN |
| easy_BMW | easy | BMW | BMW | 0.66 | confident | OK |
| small_BMW | small | BMW | BMW | 0.7474 | ambiguous | OK |
| blur_BMW | blur | BMW | BMW | 0.8393 | ambiguous | OK |
| conflict_BMW_vs_PEPSI | conflict | BMW | PEPSI | 0.3243 | confident | FP_WRONG_CANDIDATE |
| negative_NEG | negative | None | None | 0.0 | abstain | OK |
