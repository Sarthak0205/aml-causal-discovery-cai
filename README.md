# Additive Noise Model Causal Discovery with Causal Feature Attribution

## Project Overview

This AML mini project implements a causal discovery and explainability pipeline based on the Additive Noise Model (ANM).

The project first discovers a directed acyclic graph (DAG) from observational data using pairwise ANM causal discovery. It then fits the discovered causal structure and uses Causal Feature Attribution (CAI) to estimate the local causal contribution of individual features to a prediction made by a user-provided machine learning model.

The predictive model is loaded from an external `.pkl` file, making the CAI stage model-agnostic. The supplied model only needs to expose a compatible `predict()` interface, with `predict_proba()` used when available for classification probability explanations.

## Methodology

The complete pipeline is:

1. Dataset ingestion
2. Numeric feature validation
3. Median imputation
4. StandardScaler preprocessing
5. Pairwise ANM causal discovery
6. Gaussian Process regression with cross-fitted residuals
7. HSIC-based independence testing
8. Dynamic percentile confidence thresholding
9. Cycle detection and confidence-aware pruning
10. DAG validation and topological ordering
11. Predictive model loading from `model.pkl`
12. Fitting the causal ANM structural equations
13. Causal Feature Attribution using intervention sampling
14. 95% confidence intervals
15. Monte-Carlo robustness analysis
16. Visualization and interpretation

## Additive Noise Model

For a pair of variables X and Y, ANM considers the two possible directions:

    Y = f(X) + R_f

and

    X = g(Y) + R_r

where R_f and R_r are residuals.

For each direction, the project measures dependence between the proposed cause and its residual using the Hilbert-Schmidt Independence Criterion (HSIC).

The direction with the lower HSIC value is selected as the preferred causal direction.

The directional confidence margin is:

    Delta = |HSIC_forward - HSIC_reverse|

Larger values indicate a stronger separation between the two directional hypotheses.

## Dynamic Confidence Threshold

Instead of using one fixed confidence margin for every dataset, the project computes the threshold from the observed pairwise confidence margins.

For percentile p:

    tau = P_p({Delta_1, Delta_2, ..., Delta_m})

where m is the number of analyzed feature pairs.

Only edges satisfying:

    Delta >= tau

are retained.

The default configuration uses the 75th percentile.

The threshold is configurable, and a sensitivity analysis was performed across:

- 50%
- 60%
- 70%
- 75%
- 80%
- 90%

The synthetic benchmark produced:

| Percentile | Precision | Recall | F1 | Direction Accuracy | Retained Edges |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.6200 ± 0.1135 | 0.7750 ± 0.1419 | 0.6889 ± 0.1261 | 1.0000 | 5 |
| 60 | 0.6250 ± 0.1318 | 0.6250 ± 0.1318 | 0.6250 ± 0.1318 | 1.0000 | 4 |
| 70 | 0.6333 ± 0.1054 | 0.4750 ± 0.0791 | 0.5429 ± 0.0904 | 1.0000 | 3 |
| 75 | 0.6333 ± 0.1054 | 0.4750 ± 0.0791 | 0.5429 ± 0.0904 | 1.0000 | 3 |
| 80 | 0.6000 ± 0.2108 | 0.3000 ± 0.1054 | 0.4000 ± 0.1405 | 1.0000 | 2 |
| 90 | 0.9000 ± 0.3162 | 0.2250 ± 0.0791 | 0.3600 ± 0.1265 | 1.0000 | 1 |

The 50th percentile produced the highest F1 on the synthetic benchmark, while also retaining a denser graph and more transitive false positives.

The project therefore retains the 75th percentile as a conservative sparsity/precision heuristic, rather than claiming that 75% is theoretically optimal.

## Causal Graph Construction

After pairwise direction discovery:

- Low-confidence edges are removed using the dynamic threshold.
- Directed cycles are detected.
- When a cycle is found, the weakest-confidence edge is removed.
- The resulting graph is validated as a DAG.
- A topological ordering is generated.

This ensures that the graph used for downstream causal intervention sampling is acyclic.

### Important methodological limitation

The pairwise ANM stage evaluates variables two at a time and does not condition on all remaining variables.

Therefore, the resulting graph should be interpreted as an:

**ANM-based causal structure hypothesis**

rather than a guaranteed ground-truth direct causal graph.

Indirect relationships, confounding, finite-sample effects, and model misspecification can still affect the discovered structure.

## Causal Feature Attribution

The project reuses the causal ANM implementation from the previous FYP and integrates it into the AML pipeline.

For an observation x, the causal attribution of feature j is estimated as:

    CAI_j = E[f(X) | do(X_j=x_j)] - E[f(X)]

where:

- f(X) is the predictive model output,
- E[f(X)] is the baseline prediction under observational sampling,
- do(X_j=x_j) fixes feature j to its observed value and samples the remaining variables through the learned causal ANM.

A positive CAI indicates that the intervention increases the explained model output relative to the baseline.

A negative CAI indicates that the intervention decreases it.

The attribution is local to the selected observation and is not a global feature-importance measure.

## Model-Agnostic Predictive Model

The predictive model is supplied externally as:

    model.pkl

The notebook loads the model using `joblib`.

The CAI implementation does not depend on a specific machine learning algorithm.

The model is adapted through a scalar prediction function:

    prediction_function(X)

For classification models with `predict_proba()`, the probability corresponding to the predicted class is explained.

For models without `predict_proba()`, the direct prediction output is used.

The CAI interface was explicitly tested with:

- Logistic Regression
- Random Forest Classifier

This demonstrates that the CAI stage is model-agnostic for compatible predictive models.

### Model compatibility requirement

The supplied `.pkl` model must expect the same feature representation provided by the notebook.

In the current implementation, the causal features passed to the predictive model are the StandardScaler-transformed features.

For a more general deployment scenario, the recommended approach is to save preprocessing and the predictive model together as a single sklearn Pipeline.

The notebook does not retrain the supplied `.pkl` model. It loads the saved model and uses it for prediction and causal attribution.

## Predictive Validation

A separate 5-fold stratified cross-validation diagnostic was performed using the Wine classification dataset.

Results:

- Mean accuracy: **0.9833**
- Standard deviation: **0.0136**
- Fold accuracies:
  - 0.9722
  - 0.9722
  - 0.9722
  - 1.0000
  - 1.0000

Training accuracy of the fitted demonstration model was 1.0000.

Training accuracy is reported only as a diagnostic; the cross-validation result is the more appropriate estimate of predictive performance.

## Dataset 1: Wine

The Wine dataset contains:

- 178 observations
- 13 numeric causal features
- 3 prediction classes

The CSV used by the project is:

    wine.csv

The CSV contains the 13 numeric features.

For the Wine demonstration, the official target labels are recovered from the verified sklearn Wine reference dataset after confirming that the CSV feature values match the reference features.

### Final discovered DAG

The 75th-percentile threshold produced 20 retained edges.

| # | Direction | Confidence |
|---:|---|---:|
| 1 | proline → total_phenols | 0.011298 |
| 2 | proline → od280/od315_of_diluted_wines | 0.008469 |
| 3 | malic_acid → hue | 0.007286 |
| 4 | proline → flavanoids | 0.007085 |
| 5 | proline → proanthocyanins | 0.006264 |
| 6 | malic_acid → total_phenols | 0.005834 |
| 7 | proline → alcalinity_of_ash | 0.004901 |
| 8 | proline → alcohol | 0.004785 |
| 9 | proline → magnesium | 0.004274 |
| 10 | color_intensity → alcalinity_of_ash | 0.004190 |
| 11 | color_intensity → total_phenols | 0.004181 |
| 12 | malic_acid → alcalinity_of_ash | 0.004144 |
| 13 | flavanoids → alcohol | 0.003852 |
| 14 | od280/od315_of_diluted_wines → proanthocyanins | 0.003753 |
| 15 | malic_acid → proline | 0.003655 |
| 16 | proline → nonflavanoid_phenols | 0.003527 |
| 17 | flavanoids → od280/od315_of_diluted_wines | 0.003405 |
| 18 | od280/od315_of_diluted_wines → color_intensity | 0.003330 |
| 19 | nonflavanoid_phenols → alcalinity_of_ash | 0.002972 |
| 20 | flavanoids → total_phenols | 0.002933 |

The resulting graph was validated as a DAG and contained no cycles.

## Wine CAI Result

CAI was computed for observation 0.

The predictive model classified the observation as class 0, and the explained output was:

**Probability of class 0**

The strongest normalized causal attributions were:

| Feature | Normalized CAI |
|---|---:|
| proline | 0.375056 |
| malic_acid | 0.145373 |
| alcohol | 0.134977 |
| flavanoids | 0.121194 |
| od280/od315_of_diluted_wines | 0.088419 |
| alcalinity_of_ash | 0.058340 |
| color_intensity | 0.025643 |
| nonflavanoid_phenols | 0.020759 |
| magnesium | 0.013450 |
| total_phenols | 0.008432 |
| proanthocyanins | 0.004101 |
| ash | 0.003173 |
| hue | -0.001083 |

The strongest positive attribution was proline.

The largest negative attribution was hue, although its confidence interval crossed zero.

### 95% confidence intervals

| Feature | CAI | 95% CI |
|---|---:|---:|
| proline | 0.542078 | [0.512411, 0.571745] |
| malic_acid | 0.210111 | [0.174077, 0.246145] |
| alcohol | 0.195086 | [0.159631, 0.230540] |
| flavanoids | 0.175165 | [0.139274, 0.211057] |
| od280/od315_of_diluted_wines | 0.127793 | [0.092274, 0.163313] |
| alcalinity_of_ash | 0.084320 | [0.048876, 0.119764] |
| color_intensity | 0.037062 | [0.001504, 0.072621] |
| nonflavanoid_phenols | 0.030004 | [-0.005315, 0.065323] |
| magnesium | 0.019440 | [-0.015833, 0.054713] |
| total_phenols | 0.012186 | [-0.023014, 0.047387] |
| proanthocyanins | 0.005927 | [-0.029152, 0.041007] |
| ash | 0.004586 | [-0.030398, 0.039569] |
| hue | -0.001565 | [-0.036569, 0.033439] |

Confidence intervals crossing zero indicate weaker statistical evidence for a non-zero local attribution.

## CAI Monte-Carlo Robustness

CAI was evaluated using two Monte-Carlo sample sizes:

- K=1000
- K=2000

The same observation and explained class were used in both runs.

Results:

- Baseline prediction at K=1000: **0.311357**
- Baseline prediction at K=2000: **0.305126**
- Spearman rank correlation between attribution rankings: **0.9066**

The seven highest-ranked features retained the same ordering:

1. proline
2. malic_acid
3. alcohol
4. flavanoids
5. od280/od315_of_diluted_wines
6. alcalinity_of_ash
7. color_intensity

This supports the numerical robustness of the reported top CAI rankings as the Monte-Carlo sample size increases.

## Dataset 2: Breast Cancer Wisconsin Diagnostic

The project also supports the Breast Cancer Wisconsin Diagnostic feature dataset:

- 569 observations
- 30 numeric features
- No causal ground-truth graph is assumed.

The CSV is:

    breast_cancer.csv

Using the 75th-percentile confidence threshold:

- Pairwise comparisons: 435
- Retained edges: 109
- Cycles after pruning: none
- Final graph: valid DAG

Six features were isolated in the final graph:

- mean fractal dimension
- texture error
- smoothness error
- symmetry error
- worst symmetry
- worst fractal dimension

The resulting graph is treated as an observational ANM-based causal structure hypothesis rather than a verified ground-truth causal graph.

## Reproducibility

The project uses a fixed random seed:

    RANDOM_STATE = 42

Other important configuration parameters include:

    N_SPLITS = 5
    GP_ALPHA = 1e-3
    DYNAMIC_PERCENTILE = 75.0
    SUBSAMPLE_SIZE = 500
    BW_SUBSAMPLE_SIZE = 300
    CAI_SAMPLES = 1000

The ANM pairwise implementation was also tested for deterministic behavior and global NumPy random-state isolation.

## Testing

The project contains three test suites.

### ANM Pairwise Tests

`test_anm_pairwise.py`

**6/6 tests passed**

Tests include:

- Basic direction output and structure
- Determinism
- Strict input validation
- Controlled subsampling and seed sensitivity
- Known synthetic ANM orientation
- Global NumPy RNG isolation

### Validation Tests

`test_validation.py`

**12/12 tests passed**

Tests include:

- Deterministic benchmark generation
- Synthetic benchmark schema
- Ground-truth DAG
- Dataset mode handling
- External CSV anti-heuristic validation
- Sensitivity result encapsulation
- Missing-file handling
- Graph evaluation and false-positive breakdown
- Minimum-confidence cycle elimination
- Multi-seed benchmark invariance
- Pipeline execution and multi-seed integrity
- Multi-seed percentile sensitivity structure and determinism

### CAI Tests

`test_cai.py`

**7/7 tests passed**

Tests include:

- CAI output dimensions
- Numerical finiteness
- Intervention correctness
- Determinism
- Normalized attribution integrity
- Confidence interval consistency
- Model-agnostic prediction interface using Random Forest

### Overall

**25/25 automated tests passed.**

## Project Structure

    AML/
    ├── ANM_DAG_Discovery.ipynb
    ├── anm_pairwise.py
    ├── cai.py
    ├── generate_benchmark.py
    ├── server.py
    ├── web/
    │   ├── index.html
    │   ├── style.css
    │   └── app.js
    ├── test_anm_pairwise.py
    ├── test_cai.py
    ├── test_validation.py
    ├── wine.csv
    ├── model.pkl
    ├── requirements.txt
    ├── Div-B_Group 12_AML_Research_Paper.pdf
    ├── README.md
    └── .gitignore

The local `.venv` directory is intentionally excluded from the repository.

## Installation

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Running the Web Demonstration

Start the local web server:

    python3 server.py 8000

Open your browser and navigate to:

    http://localhost:8000/

The web interface demonstrates the complete research pipeline:
- An interactive 8-step methodology flow
- Topological role assignment and SVG visualization of the 20-edge discovered DAG
- Local interventional Causal Feature Attribution (CAI) with 95% confidence intervals explaining Observation #0 ($P(\text{class } 0)$)
- An interactive, sortable, and filterable table of all 78 evaluated variable pairs

## Running the Notebook

Launch Jupyter:

    jupyter notebook

Open:

    ANM_DAG_Discovery.ipynb

Run the notebook from the first cell using a clean kernel.

The default configuration uses:

    DATASET_PATH = "wine.csv"
    MODEL_PATH = "model.pkl"

## Running the Tests

Run the three automated test suites:

    python3 test_anm_pairwise.py
    python3 test_validation.py
    python3 test_cai.py

Expected result:

    25/25 tests passed

## Limitations

The following limitations should be considered when interpreting the results:

1. Pairwise causal discovery does not condition on all other variables.
2. Observational data alone does not guarantee identification of the true causal DAG.
3. ANM assumptions may not hold for every real-world variable pair.
4. Pairwise relationships can include indirect causal associations.
5. The percentile threshold is a heuristic and is not theoretically universal.
6. The CAI result depends on the discovered DAG and fitted structural equations.
7. CAI is local to the selected observation and explained model output.
8. Monte-Carlo estimates introduce sampling variability.
9. A compatible preprocessing representation is required by the supplied predictive model.
10. CAI should not be interpreted as definitive real-world intervention evidence without stronger causal assumptions and experimental or domain validation.

## Academic Interpretation

The project should be presented as a computational framework for:

**ANM-based causal structure discovery followed by local causal feature attribution of a predictive model.**

The causal graph is a data-driven hypothesis generated from observational relationships.

The CAI stage estimates model-output changes under interventions propagated through the learned causal ANM.

Therefore, the framework combines:

- Causal discovery
- Causal intervention
- Machine learning prediction
- Local explainability
- Uncertainty estimation
- Robustness analysis

without claiming that the discovered observational graph is automatically the true causal graph.

## Requirements

The main Python dependencies are pinned in `requirements.txt`.

The project requires:

- Python 3
- NumPy
- Pandas
- SciPy
- scikit-learn
- NetworkX
- Matplotlib
- Jupyter
- IPykernel
- Joblib
