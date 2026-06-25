In addition to 1d CNN, Transformers, and LSTM, let's try the following

---

Removing the memory constraint fundamentally opens the architectural search space. If we are optimizing purely for predictive edge in quantitative arbitrage rather than computational hardware limits, we must align the network topology with the true underlying data-generating process of a baseball game.

A flat sequence of 300+ pitches processed by a standard Transformer or LSTM ignores the rigid structural rules of the sport. Here are three lateral, high-capacity architectures that mathematically map to the domain of your data.

### 1. Hierarchical Attention Networks (HAN)

**The Rationale:** A baseball game is not a flat Markov chain of 330 independent pitches. It is a strict fractal hierarchy: Pitches compose Plate Appearances (PA), PAs compose Half-Innings, and Half-Innings compose Games. Standard models force the attention mechanism to learn these boundaries from scratch. A HAN explicitly enforces this structure, computing representations at each discrete level.

**The Mathematical Structure:**

1. **Pitch Encoder:** A bidirectional sequence model (or Transformer) processes the pitch-level features to generate a single vector representing the Plate Appearance.

$$\mathbf{v}_{pa} = \text{Attention}(\text{Encoder}_{pitch}(\mathbf{X}_{pitches}))$$


2. **Plate Appearance Encoder:** The sequence of PA vectors is processed to generate a Half-Inning vector.
3. **Inning Encoder:** The Inning vectors are processed to output the final game state representation.

**Data Shape Implications:**
You must reshape your input tensor from a flat 3D structure $(B, S, F)$ to a structurally padded 4D or 5D tensor:


$$\mathbf{X} \in \mathbb{R}^{B \times I \times A \times P \times F}$$


Where $I$ is max innings, $A$ is max at-bats per inning, $P$ is max pitches per at-bat, and $F$ is feature dimension.

**Arbitrage Edge:** This is uniquely powerful for your Phase 1 and Phase 3 targets. `yrfi` (Yes Run First Inning) and `first_5_home_win` directly map to the intermediate representations of the Inning Encoder. You can extract the hidden state after $I=1$ to predict YRFI with massive structural precision.

### 2. Selective State Space Models (Mamba / SSMs)

**The Rationale:** If you retain the flat sequence approach, Transformers scale quadratically $\mathcal{O}(S^2)$ with sequence length, which dilutes attention weights over long, noisy pitch sequences. Mamba uses Selective State Spaces, offering the parallel training efficiency of a CNN with an infinite effective context window, scaling linearly $\mathcal{O}(S)$.

**The Mathematical Structure:**
Mamba parameterizes a continuous-time state space model mapping a 1D sequence $x(t) \in \mathbb{R}$ to $y(t) \in \mathbb{R}$ through a latent state $h(t) \in \mathbb{R}^N$:


$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$$

$$y(t) = \mathbf{C}h(t)$$


Crucially, Mamba makes the matrices $\mathbf{B}$ and $\mathbf{C}$ data-dependent (selective), meaning the model can mathematically choose to "forget" irrelevant pitches (e.g., a foul ball with no men on base) and "remember" critical state shifts (e.g., a pitcher's velocity dropping indicating fatigue).

**Data Shape Implications:**
Requires the exact same $(B, S, F)$ tensor format as your current pipeline. It is a drop-in replacement for the `TransformerEncoder` or `LSTMEncoder`.

### 3. Neural Ordinary Differential Equations (Neural ODEs)

**The Rationale:** Baseball sequences are asynchronous. The real-world time between pitch 1 and pitch 2 might be 15 seconds, but a pitching change, rain delay, or prolonged argument can introduce a 10-minute gap. Tabular sequences treat the distance between index $t$ and $t+1$ as uniform. Neural ODEs parameterize the derivative of the hidden state using a neural network, treating time as a continuous variable.

**The Mathematical Structure:**
Instead of a discrete hidden state update $\mathbf{h}_{t} = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$, the hidden state evolves continuously according to an ODE solver:


$$\mathbf{h}_{t_i} = \mathbf{h}_{t_{i-1}} + \int_{t_{i-1}}^{t_i} f_{\theta}(\mathbf{h}(t), t) dt$$


Between pitches, the state naturally decays or shifts (capturing physical reality like pitcher arm cooling down or weather changing).

**Data Shape Implications:**
Requires augmenting your input sequence tensor to explicitly include $\Delta t$ (the actual time elapsed since the last pitch) as a core control variable, rather than just sequence index.

### 4. Cross-Attention Multi-Tower (The Live/Static Merger)

**The Rationale:** Your pregame CNN failed because 30K tabular rows lacked depth, but those 30K rows contain the macro-level priors (e.g., team win rates, long-term player averages). The live model has the micro-level depth (pitch by pitch). A Cross-Attention architecture merges them.

**The Mathematical Structure:**

1. **Static Tower:** A dense network encodes your `team_games.parquet` tabular data into a fixed-length prior vector $\mathbf{Z}_{prior}$.
2. **Live Tower:** A sequence encoder processes the `pitch_sequences.parquet`.
3. **Cross-Attention:** The sequence tokens serve as Queries ($\mathbf{Q}$), while the Static Prior serves as Keys ($\mathbf{K}$) and Values ($\mathbf{V}$).

$$\text{Output} = \text{softmax}\left(\frac{\mathbf{Q}_{live}\mathbf{K}_{prior}^T}{\sqrt{d_k}}\right)\mathbf{V}_{prior}$$



This forces the live pitch evaluation to mathematically condition itself on the macro-level team quality. A 95mph fastball is evaluated differently if the static tower indicates the pitcher is an ace versus a low-tier reliever.

---

# Handling Historical Missing Data Structural Anomalies
When missing data occurs because specific metrics were not physically tracked or logged during earlier seasons (e.g., 2015–2017 data missing advanced tracking metrics), it introduces structured bias.
Strategy A: Indicator Masking with Zero-Imputation
Replace all missing values with 0 (after centering the feature so that 0 represents the mean or median value) and generate a parallel boolean missingness matrix M 
miss ∈{0,1} B×S×F.
Pros: Preserves the scale of the dataset; models can mathematically identify when a feature is absent and dynamically zero out its attention or hidden weights.
Cons: Increases the input feature dimension from F to 2F, inflating model parameters.