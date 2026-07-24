# MLB Pregame Model Metrics Report

> **DRY-RUN — 2026-07-13**
>
> **Per-model rows** are isotonic-calibrated OOF metrics from the **prior** training run (2015–2026, 2020 excluded) — kept as baseline reference. In Ens / Weight columns reflect the **current** EC2 ensemble composition.
> **Calibration Comparison** uses a held-out LOYO set (n=21,931); all four calibration schemes (none, Platt, isotonic, temperature) × two weighting methods (slsqp, meta) shown.
> **Deployed Ensemble** metrics use the full OOF set (n=23,206) from the production ensemble on EC2.

---

## Classification Targets

**Metrics:** Log-Loss (lower=better) · AUC-ROC (higher=better) · Brier Score (lower=better) · ECE (lower=better)


### Extra Innings

| Model                  | Log-Loss | AUC-ROC | Brier   | ECE     | In Ens | Weight |
| ---------------------- | -------- | ------- | ------- | ------- | ------ | ------ |
| hist_gradient_boosting | 0.25609  | 0.5850  | 0.06753 | 0.00677 | ✓      | 0.602  |
| extra_trees            | 0.25760  | 0.5872  | 0.06762 | 0.00603 | ✓      | 0.242  |
| lightgbm               | 0.25721  | 0.5832  | 0.06764 | 0.00834 | ✓      | 0.073  |
| mlp                    | 0.29796  | 0.5735  | 0.07187 | 0.04888 | ✓      | 0.069  |
| random_forest          | 0.25657  | 0.5790  | 0.06766 | 0.01060 | ✓      | 0.014  |
| catboost               | 0.25750  | 0.5842  | 0.06760 | 0.00457 | —      | —      |
| xgboost                | 0.25662  | 0.5851  | 0.06756 | 0.00645 | ✗      | ✗      |
| adaboost               | 0.27403  | 0.5063  | 0.07022 | 0.04565 | ✗      | ✗      |

_✗ = not in current model pool (dropped from this training run)_

**Calibration Comparison** (n=21,931, best = lowest log-loss):

| Calibrator  | Weights | Log-Loss    | AUC-ROC | Brier   | ECE     |
| ----------- | ------- | ----------- | ------- | ------- | ------- |
| none (raw)  | slsqp   | 0.25587     | 0.58346 | 0.06754 | 0.00703 |
| none (raw)  | meta    | 0.25615     | 0.58730 | 0.06751 | 0.00588 |
| platt       | slsqp   | 0.25704     | 0.58684 | 0.06760 | 0.00226 |
| platt       | meta    | 0.25722     | 0.58686 | 0.06759 | 0.00279 |
| isotonic    | slsqp   | **0.25305** | 0.59946 | 0.06725 | 0.00274 |
| isotonic    | meta    | **0.25318** | 0.59966 | 0.06725 | 0.00357 |
| temperature | slsqp   | 0.25542     | 0.58335 | 0.06751 | 0.00378 |
| temperature | meta    | 0.25584     | 0.58695 | 0.06746 | 0.00164 |

**Deployed Ensemble** (n=23,206): Log-Loss=**0.25391** · AUC=**0.58852** · Brier=**0.06671** · ECE=**0.00608** · Accuracy=0.9274


### First 5 Home Win

| Model                  | Log-Loss | AUC-ROC | Brier   | ECE     | In Ens | Weight |
| ---------------------- | -------- | ------- | ------- | ------- | ------ | ------ |
| lightgbm               | 0.67812  | 0.5846  | 0.24259 | 0.00890 | ✓      | 0.479  |
| xgboost                | 0.67847  | 0.5843  | 0.24274 | 0.00559 | ✓      | 0.321  |
| random_forest          | 0.67883  | 0.5832  | 0.24291 | 0.00632 | ✓      | 0.200  |
| hist_gradient_boosting | 0.68030  | 0.5763  | 0.24365 | 0.00919 | —      | —      |
| catboost               | 0.68098  | 0.5724  | 0.24398 | 0.00718 | —      | —      |
| adaboost               | 0.68248  | 0.5697  | 0.24467 | 0.01562 | —      | —      |
| extra_trees            | 0.68280  | 0.5643  | 0.24487 | 0.00578 | —      | —      |
| mlp                    | 0.76756  | 0.5507  | 0.27522 | 0.14288 | ✗      | ✗      |

_✗ = not in current model pool (dropped from this training run)_

**Calibration Comparison** (n=21,931):

| Calibrator  | Weights | Log-Loss    | AUC-ROC | Brier       | ECE     |
| ----------- | ------- | ----------- | ------- | ----------- | ------- |
| none (raw)  | slsqp   | 0.67758     | 0.58756 | 0.24231     | 0.01055 |
| none (raw)  | meta    | 0.67758     | 0.58752 | 0.24231     | 0.01089 |
| platt       | slsqp   | 0.67750     | 0.58753 | 0.24227     | 0.00976 |
| platt       | meta    | 0.67750     | 0.58755 | 0.24227     | 0.00959 |
| isotonic    | slsqp   | **0.67619** | 0.59112 | **0.24165** | 0.00743 |
| isotonic    | meta    | **0.67619** | 0.59117 | **0.24165** | 0.00761 |
| temperature | slsqp   | 0.67753     | 0.58754 | 0.24229     | 0.00926 |
| temperature | meta    | 0.67754     | 0.58752 | 0.24229     | 0.01000 |

**Deployed Ensemble** (n=23,206): Log-Loss=**0.67720** · AUC=**0.58812** · Brier=**0.24214** · ECE=**0.00552** · Accuracy=0.5692


### Home Win

> ⚠ Major composition change from prior run: catboost (was 0.710) and lightgbm (was — prior weight) **dropped from the candidate pool**. hist_gradient_boosting + elasticnet now dominate.

| Model                  | Log-Loss | AUC-ROC | Brier   | ECE     | In Ens | Weight |
| ---------------------- | -------- | ------- | ------- | ------- | ------ | ------ |
| hist_gradient_boosting | 0.64411  | 0.6736  | 0.22645 | 0.00877 | ✓      | 0.605  |
| elasticnet             | 0.65196  | 0.6694  | 0.22758 | 0.01083 | ✓      | 0.377  |
| lda                    | 0.65626  | 0.6528  | 0.23146 | 0.01155 | ✓      | 0.010  |
| extra_trees            | 0.66651  | 0.6481  | 0.23686 | 0.04995 | ✓      | 0.007  |
| xgboost                | 0.67881  | 0.5904  | 0.24291 | 0.00678 | —      | —      |
| qda                    | 0.84524  | 0.6075  | 0.26406 | 0.13707 | —      | —      |
| knn                    | 0.67925  | 0.5893  | 0.24313 | 0.01015 | —      | —      |
| sgd                    | 0.67976  | 0.5879  | 0.24337 | 0.00514 | —      | —      |
| ridge                  | 0.68169  | 0.5942  | 0.24430 | 0.03501 | —      | —      |
| random_forest          | 0.67883  | 0.5917  | 0.24292 | 0.00842 | —      | —      |
| catboost               | 0.64335  | 0.6753  | 0.22608 | 0.01135 | ✗      | ✗      |
| lightgbm               | 0.64315  | 0.6744  | 0.22607 | 0.00882 | ✗      | ✗      |
| adaboost               | 0.64412  | 0.6735  | 0.22647 | 0.01075 | ✗      | ✗      |
| lasso                  | 0.65269  | 0.6680  | 0.22788 | 0.00905 | ✗      | ✗      |
| bagging_logreg         | 0.65430  | 0.6644  | 0.22876 | 0.00998 | ✗      | ✗      |
| gaussian_nb            | 0.69116  | 0.6264  | 0.23933 | 0.02204 | ✗      | ✗      |
| mlp                    | 0.67046  | 0.6256  | 0.23788 | 0.02345 | ✗      | ✗      |
| logistic_regression    | 0.67761  | 0.5941  | 0.24235 | 0.00419 | ✗      | ✗      |

_✗ = not in current model pool (dropped from this training run)_

**Calibration Comparison** (n=21,931):

| Calibrator  | Weights | Log-Loss    | AUC-ROC     | Brier       | ECE         |
| ----------- | ------- | ----------- | ----------- | ----------- | ----------- |
| none (raw)  | slsqp   | 0.64269     | 0.67558     | 0.22585     | 0.01040     |
| none (raw)  | meta    | 0.64269     | 0.67559     | 0.22585     | 0.01080     |
| platt       | slsqp   | 0.64253     | 0.67555     | 0.22579     | 0.00922     |
| platt       | meta    | 0.64253     | 0.67556     | 0.22579     | 0.00888     |
| isotonic    | slsqp   | **0.64032** | **0.67778** | **0.22492** | 0.00653     |
| isotonic    | meta    | **0.64032** | **0.67783** | **0.22492** | **0.00580** |
| temperature | slsqp   | 0.64262     | 0.67554     | 0.22582     | 0.00901     |
| temperature | meta    | 0.64262     | 0.67557     | 0.22582     | 0.00919     |

**Deployed Ensemble** (n=23,206): Log-Loss=**0.64145** · AUC=**0.67777** · Brier=**0.22526** · ECE=**0.01082** · Accuracy=0.6269


### Yrfi

| Model                  | Log-Loss | AUC-ROC | Brier   | ECE     | In Ens | Weight |
| ---------------------- | -------- | ------- | ------- | ------- | ------ | ------ |
| catboost               | 0.69258  | 0.5208  | 0.24972 | 0.00576 | ✓      | 0.364  |
| random_forest          | 0.69252  | 0.5225  | 0.24969 | 0.00906 | ✓      | 0.331  |
| adaboost               | 0.69309  | 0.5161  | 0.24997 | 0.00942 | ✓      | 0.295  |
| extra_trees            | 0.69257  | 0.5198  | 0.24971 | 0.00648 | ✓      | 0.010  |
| hist_gradient_boosting | 0.69313  | 0.5142  | 0.24999 | 0.00928 | —      | —      |
| lightgbm               | 0.69289  | 0.5148  | 0.24987 | 0.00635 | —      | —      |
| xgboost                | 0.69298  | 0.5178  | 0.24991 | 0.00842 | —      | —      |
| mlp                    | 0.81147  | 0.4956  | 0.29467 | 0.17645 | ✗      | ✗      |

_✗ = not in current model pool (dropped from this training run)_

**Calibration Comparison** (n=21,931):

| Calibrator  | Weights | Log-Loss    | AUC-ROC     | Brier       | ECE         |
| ----------- | ------- | ----------- | ----------- | ----------- | ----------- |
| none (raw)  | slsqp   | 0.69245     | 0.52257     | 0.24965     | 0.00659     |
| none (raw)  | meta    | 0.69247     | 0.52297     | 0.24966     | 0.00804     |
| platt       | slsqp   | 0.69235     | 0.52278     | 0.24960     | 0.00059     |
| platt       | meta    | 0.69235     | 0.52278     | 0.24961     | 0.00059     |
| isotonic    | slsqp   | **0.69177** | **0.52776** | **0.24932** | 0.00211     |
| isotonic    | meta    | **0.69177** | **0.52783** | **0.24932** | **0.00153** |
| temperature | slsqp   | 0.69244     | 0.52304     | 0.24965     | 0.00606     |
| temperature | meta    | 0.69246     | 0.52255     | 0.24966     | 0.00648     |

**Deployed Ensemble** (n=23,206): Log-Loss=**0.69259** · AUC=**0.52111** · Brier=**0.24972** · ECE=**0.00775** · Accuracy=0.5147


---

## Regression Targets

**Metrics:** MAE (lower=better) · RMSE (lower=better) · R² (higher=better) · Huber Loss δ=1.35 (lower=better)

_No calibration grid for regression targets. Weighting comparison (slsqp vs meta) shown from n=21,931 held-out set._


### Away Runs

> ⚠ Composition change: sgd (was 0.284 weight) dropped. lightgbm now 0.976 — near-singleton ensemble.

| Model                  | MAE    | RMSE   | R²      | Huber  | In Ens | Weight |
| ---------------------- | ------ | ------ | ------- | ------ | ------ | ------ |
| lightgbm               | 2.5706 | 3.4297 | 0.0084  | 2.6607 | ✓      | 0.976  |
| mlp                    | 2.6934 | 3.4850 | -0.0238 | 2.8194 | ✓      | 0.024  |
| catboost               | 2.5714 | 3.4352 | 0.0053  | 2.6619 | ✗      | ✗      |
| sgd                    | 2.5779 | 3.4374 | 0.0040  | 2.6699 | ✗      | ✗      |
| hist_gradient_boosting | 2.6130 | 3.3896 | 0.0315  | 2.7119 | —      | —      |
| extra_trees            | 2.6134 | 3.3915 | 0.0304  | 2.7118 | —      | —      |
| random_forest          | 2.6216 | 3.3961 | 0.0278  | 2.7218 | —      | —      |
| elasticnet             | 2.6235 | 3.3976 | 0.0269  | 2.7238 | —      | —      |
| lasso                  | 2.6235 | 3.3974 | 0.0270  | 2.7236 | —      | —      |
| gaussian_nb            | 2.6240 | 3.3979 | 0.0267  | 2.7245 | —      | —      |
| lda                    | 2.6240 | 3.3979 | 0.0267  | 2.7245 | —      | —      |
| logistic_regression    | 2.6240 | 3.3980 | 0.0267  | 2.7245 | —      | —      |
| qda                    | 2.6240 | 3.3979 | 0.0267  | 2.7245 | —      | —      |
| ridge                  | 2.6240 | 3.3980 | 0.0267  | 2.7245 | —      | —      |
| bagging_logreg         | 2.6247 | 3.3983 | 0.0265  | 2.7253 | —      | —      |
| adaboost               | 2.6252 | 3.4033 | 0.0236  | 2.7298 | —      | —      |
| knn                    | 2.6256 | 3.4096 | 0.0200  | 2.7279 | —      | —      |

**Weighting Comparison** (n=21,931):

| Weights | MAE    | RMSE   | R²     | Huber  |
| ------- | ------ | ------ | ------ | ------ |
| slsqp   | 2.5705 | 3.4271 | 0.0099 | 2.8700 |
| meta    | 2.5963 | 3.3864 | 0.0333 | 2.9018 |

**Deployed Ensemble** (n=23,206): MAE=**2.578** · RMSE=**3.441** · R²=**0.0104** · Huber=**2.8807**


### First 5 Home Run Diff

| Model                  | MAE    | RMSE   | R²      | Huber  | In Ens | Weight |
| ---------------------- | ------ | ------ | ------- | ------ | ------ | ------ |
| hist_gradient_boosting | 2.5977 | 3.4563 | 0.0414  | 2.7121 | ✓      | 0.348  |
| sgd                    | 2.6106 | 3.4704 | 0.0335  | 2.7297 | ✓      | 0.234  |
| lightgbm               | 2.6027 | 3.4589 | 0.0400  | 2.7184 | ✓      | 0.219  |
| random_forest          | 2.5995 | 3.4545 | 0.0424  | 2.7131 | ✓      | 0.107  |
| mlp                    | 2.7199 | 3.5718 | -0.0238 | 2.8657 | ✓      | 0.092  |
| extra_trees            | 2.6027 | 3.4616 | 0.0385  | 2.7185 | —      | —      |
| catboost               | 2.6104 | 3.4717 | 0.0328  | 2.7282 | —      | —      |
| knn                    | 2.6322 | 3.4972 | 0.0185  | 2.7570 | —      | —      |
| adaboost               | 2.6114 | 3.4738 | 0.0317  | 2.7317 | —      | —      |
| lasso                  | 2.6096 | 3.4642 | 0.0370  | 2.7275 | ✗      | ✗      |
| elasticnet             | 2.6098 | 3.4642 | 0.0370  | 2.7277 | ✗      | ✗      |
| gaussian_nb            | 2.6102 | 3.4643 | 0.0369  | 2.7281 | ✗      | ✗      |
| lda                    | 2.6102 | 3.4643 | 0.0369  | 2.7281 | ✗      | ✗      |
| qda                    | 2.6102 | 3.4643 | 0.0369  | 2.7281 | ✗      | ✗      |
| logistic_regression    | 2.6106 | 3.4643 | 0.0370  | 2.7285 | ✗      | ✗      |
| ridge                  | 2.6106 | 3.4643 | 0.0370  | 2.7285 | ✗      | ✗      |
| bagging_logreg         | 2.6108 | 3.4645 | 0.0369  | 2.7287 | ✗      | ✗      |

**Weighting Comparison** (n=21,931):

| Weights | MAE    | RMSE   | R²     | Huber  |
| ------- | ------ | ------ | ------ | ------ |
| slsqp   | 2.5917 | 3.4477 | 0.0461 | 2.9212 |
| meta    | 2.5920 | 3.4483 | 0.0458 | 2.9220 |

**Deployed Ensemble** (n=23,206): MAE=**2.600** · RMSE=**3.465** · R²=**0.0474** · Huber=**2.9336**


### First 5 Total Runs

| Model                  | MAE    | RMSE   | R²      | Huber  | In Ens | Weight |
| ---------------------- | ------ | ------ | ------- | ------ | ------ | ------ |
| catboost               | 2.7365 | 3.5958 | 0.0015  | 2.8767 | ✓      | 0.535  |
| lightgbm               | 2.7367 | 3.5911 | 0.0041  | 2.8767 | ✓      | 0.436  |
| knn                    | 2.8144 | 3.6086 | -0.0057 | 2.9799 | ✓      | 0.029  |
| sgd                    | 2.7523 | 3.6144 | -0.0089 | 2.9066 | —      | —      |
| extra_trees            | 2.7760 | 3.5627 | 0.0198  | 2.9259 | —      | —      |
| random_forest          | 2.7821 | 3.5650 | 0.0185  | 2.9338 | —      | —      |
| hist_gradient_boosting | 2.7846 | 3.5653 | 0.0183  | 2.9383 | —      | —      |
| adaboost               | 2.7905 | 3.5730 | 0.0141  | 2.9475 | —      | —      |
| gaussian_nb            | 2.8082 | 3.5935 | 0.0027  | 2.9639 | —      | —      |
| lda                    | 2.8082 | 3.5935 | 0.0027  | 2.9639 | —      | —      |
| logistic_regression    | 2.8082 | 3.5935 | 0.0027  | 2.9639 | —      | —      |
| qda                    | 2.8082 | 3.5935 | 0.0027  | 2.9639 | —      | —      |
| ridge                  | 2.8083 | 3.5935 | 0.0027  | 2.9639 | —      | —      |
| bagging_logreg         | 2.8086 | 3.5937 | 0.0026  | 2.9642 | —      | —      |
| lasso                  | 2.8104 | 3.5943 | 0.0023  | 2.9641 | —      | —      |
| elasticnet             | 2.8109 | 3.5945 | 0.0022  | 2.9644 | —      | —      |
| mlp                    | 2.8766 | 3.7000 | -0.0573 | 3.0616 | —      | —      |

**Weighting Comparison** (n=21,931):

| Weights | MAE    | RMSE   | R²     | Huber  |
| ------- | ------ | ------ | ------ | ------ |
| slsqp   | 2.7356 | 3.5902 | 0.0046 | 3.1069 |
| meta    | 2.7548 | 3.5587 | 0.0220 | 3.1325 |

**Deployed Ensemble** (n=23,206): MAE=**2.750** · RMSE=**3.626** · R²=**0.0040** · Huber=**3.1280**


### Home Run Diff

> ⚠ Composition change: bagging_logreg (was 0.997) replaced by sgd (0.989). Now 2-member ensemble.

| Model               | MAE    | RMSE   | R²      | Huber  | In Ens | Weight |
| ------------------- | ------ | ------ | ------- | ------ | ------ | ------ |
| sgd                 | 3.3792 | 4.4124 | 0.0910  | 3.7283 | ✓      | 0.989  |
| mlp                 | 3.8593 | 4.9765 | -0.1563 | 4.3700 | ✓      | 0.011  |
| logistic_regression | 3.3582 | 4.3867 | 0.1015  | 3.6997 | ✗      | ✗      |
| ridge               | 3.3583 | 4.3867 | 0.1015  | 3.6998 | ✗      | ✗      |
| bagging_logreg      | 3.3586 | 4.3872 | 0.1013  | 3.7001 | ✗      | ✗      |
| lasso               | 3.3615 | 4.3879 | 0.1011  | 3.7035 | ✗      | ✗      |
| elasticnet          | 3.3620 | 4.3881 | 0.1010  | 3.7041 | ✗      | ✗      |
| gaussian_nb         | 3.3723 | 4.3966 | 0.0975  | 3.7162 | ✗      | ✗      |
| lda                 | 3.3723 | 4.3966 | 0.0975  | 3.7162 | ✗      | ✗      |
| qda                 | 3.3723 | 4.3966 | 0.0975  | 3.7162 | ✗      | ✗      |
| random_forest       | 3.4933 | 4.5340 | 0.0402  | 3.8743 | ✗      | ✗      |
| lightgbm            | 3.4955 | 4.5528 | 0.0322  | 3.8838 | ✗      | ✗      |
| catboost            | 3.4958 | 4.5499 | 0.0334  | 3.8828 | ✗      | ✗      |
| hist_gradient_boosting | 3.4969 | 4.5373 | 0.0388 | 3.8795 | ✗     | ✗      |
| extra_trees         | 3.4976 | 4.5381 | 0.0384  | 3.8780 | ✗      | ✗      |
| adaboost            | 3.5011 | 4.5467 | 0.0348  | 3.8890 | ✗      | ✗      |
| knn                 | 3.5278 | 4.5736 | 0.0233  | 3.9189 | ✗      | ✗      |

**Weighting Comparison** (n=21,931):

| Weights | MAE    | RMSE   | R²     | Huber  |
| ------- | ------ | ------ | ------ | ------ |
| slsqp   | 3.3792 | 4.4122 | 0.0911 | 4.0502 |
| meta    | 3.3793 | 4.4121 | 0.0911 | 4.0502 |

**Deployed Ensemble** (n=23,206): MAE=**3.365** · RMSE=**4.401** · R²=**0.1032** · Huber=**4.029**


### Home Runs

> ⚠ Composition change: lightgbm (was 0.917) replaced by catboost (0.955). sgd dropped.

| Model                  | MAE    | RMSE   | R²      | Huber  | In Ens | Weight |
| ---------------------- | ------ | ------ | ------- | ------ | ------ | ------ |
| catboost               | 2.4738 | 3.2872 | 0.0038  | 2.5364 | ✓      | 0.955  |
| mlp                    | 2.5951 | 3.3638 | -0.0432 | 2.6900 | ✓      | 0.045  |
| lightgbm               | 2.4756 | 3.2819 | 0.0070  | 2.5379 | ✗      | ✗      |
| sgd                    | 2.4969 | 3.3428 | -0.0302 | 2.5773 | ✗      | ✗      |
| extra_trees            | 2.5117 | 3.2510 | 0.0256  | 2.5811 | —      | —      |
| hist_gradient_boosting | 2.5199 | 3.2563 | 0.0225  | 2.5904 | —      | —      |
| random_forest          | 2.5205 | 3.2576 | 0.0216  | 2.5918 | —      | —      |
| adaboost               | 2.5259 | 3.2638 | 0.0179  | 2.5992 | —      | —      |
| gaussian_nb            | 2.5514 | 3.2918 | 0.0010  | 2.6315 | —      | —      |
| lda                    | 2.5514 | 3.2918 | 0.0010  | 2.6315 | —      | —      |
| logistic_regression    | 2.5514 | 3.2918 | 0.0010  | 2.6315 | —      | —      |
| qda                    | 2.5514 | 3.2918 | 0.0010  | 2.6315 | —      | —      |
| ridge                  | 2.5514 | 3.2918 | 0.0010  | 2.6315 | —      | —      |
| bagging_logreg         | 2.5515 | 3.2918 | 0.0010  | 2.6316 | —      | —      |
| elasticnet             | 2.5515 | 3.2918 | 0.0010  | 2.6315 | —      | —      |
| lasso                  | 2.5515 | 3.2918 | 0.0010  | 2.6315 | —      | —      |
| knn                    | 2.5714 | 3.3134 | -0.0122 | 2.6583 | —      | —      |

**Weighting Comparison** (n=21,931):

| Weights | MAE    | RMSE   | R²     | Huber  |
| ------- | ------ | ------ | ------ | ------ |
| slsqp   | 2.4734 | 3.2832 | 0.0062 | 2.7327 |
| meta    | 2.4888 | 3.2524 | 0.0248 | 2.7503 |

**Deployed Ensemble** (n=23,206): MAE=**2.482** · RMSE=**3.297** · R²=**0.0080** · Huber=**2.7445**


### Total Runs

| Model                  | MAE    | RMSE   | R²      | Huber  | In Ens | Weight |
| ---------------------- | ------ | ------ | ------- | ------ | ------ | ------ |
| lightgbm               | 3.7020 | 4.8648 | 0.0139  | 4.1566 | ✓      | 0.662  |
| catboost               | 3.7145 | 4.8908 | 0.0034  | 4.1742 | ✓      | 0.231  |
| extra_trees            | 3.7356 | 4.8268 | 0.0293  | 4.2025 | ✓      | 0.064  |
| knn                    | 3.8087 | 4.9049 | -0.0024 | 4.2974 | ✓      | 0.043  |
| sgd                    | 3.7371 | 4.9133 | -0.0058 | 4.2158 | —      | —      |
| hist_gradient_boosting | 3.7465 | 4.8340 | 0.0264  | 4.2172 | —      | —      |
| random_forest          | 3.7467 | 4.8401 | 0.0240  | 4.2174 | —      | —      |
| adaboost               | 3.7589 | 4.8463 | 0.0214  | 4.2302 | —      | —      |
| gaussian_nb            | 3.7888 | 4.8938 | 0.0022  | 4.2728 | —      | —      |
| lda                    | 3.7888 | 4.8938 | 0.0022  | 4.2728 | —      | —      |
| logistic_regression    | 3.7888 | 4.8938 | 0.0022  | 4.2728 | —      | —      |
| qda                    | 3.7888 | 4.8938 | 0.0022  | 4.2728 | —      | —      |
| ridge                  | 3.7888 | 4.8938 | 0.0022  | 4.2727 | —      | —      |
| bagging_logreg         | 3.7891 | 4.8938 | 0.0021  | 4.2732 | —      | —      |
| lasso                  | 3.7895 | 4.8948 | 0.0017  | 4.2726 | —      | —      |
| elasticnet             | 3.7916 | 4.8957 | 0.0014  | 4.2738 | —      | —      |
| mlp                    | 3.9036 | 5.0436 | -0.0599 | 4.4254 | —      | —      |

**Weighting Comparison** (n=21,931):

| Weights | MAE    | RMSE   | R²     | Huber  |
| ------- | ------ | ------ | ------ | ------ |
| slsqp   | 3.7002 | 4.8556 | 0.0177 | 4.5214 |
| meta    | 3.7203 | 4.8226 | 0.0310 | 4.5521 |

**Deployed Ensemble** (n=23,206): MAE=**3.712** · RMSE=**4.879** · R²=**0.0186** · Huber=**4.5393**
