# Cache Prefetcher

Predicting a program's next memory reference from the trace of addresses it has already
touched. If the next access can be predicted, it can be prefetched into cache before the
program asks for it — turning a cache miss into a hit.

The model is a two-layer LSTM in PyTorch, trained on windows of consecutive memory addresses.

## The data

Three CSV files of raw hexadecimal memory addresses, one per line, no header:

| File | Rows | Contents |
| --- | --- | --- |
| `memrefs_train_andor_validate.csv` | 400,000 | Address trace used for training and validation |
| `memrefs_testing.csv` | 49,999 | Held-out address trace |
| `test_targets.csv` | 50 | Ground-truth next-address for each test sequence |

```
0xbfb22b18
0xbfb22b14
0xbfb22b10
```

## Approach

### 1. Feature engineering

Each address is converted from hex to decimal, and the **delta** from the previous address is
computed alongside it. Deltas matter because access patterns are usually relative — a stride
through an array looks like a constant delta regardless of where the array lives in memory.

```
addrs        decimal_vals    deltas
0x824765c    136607324            0
0x8247620    136607264          -60
0x8247624    136607268            4
0x8247628    136607272            4
0x8247630    136607280            8
```

### 2. K-means over the address space

Addresses are clustered with **K-means (k = 6)** over the unique decimal values, and each
address is tagged with its cluster id. This acts as a cheap learned embedding: it separates
the distinct memory regions a program uses — stack, heap, code — into a small categorical
feature, since the raw 32-bit values are far too sparse to learn from directly.

### 3. Sequence framing

The trace is cut into windows of **999 consecutive addresses**, with the **1000th address as
the target**. The training trace yields 400 such sequences, split 80/20 into 320 training and
80 validation sequences.

### 4. Normalisation

Both `MinMaxScaler` and `StandardScaler` are fitted; the StandardScaler outputs feed the model.
The target is scaled with its own scaler so predictions can be inverse-transformed back into
address space.

### 5. Model

```
myLSTM(
  (lstm): LSTM(1, 64, num_layers=2, batch_first=True)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)
```

A two-layer LSTM over the sequence, followed by a linear head. Only the final timestep's output
is kept (`out[:, -1, :]`), giving one predicted address per window. Hidden and cell states are
zero-initialised per batch.

| Hyperparameter | Value |
| --- | --- |
| Hidden units | 64 |
| LSTM layers | 2 |
| Input size | 1 |
| Batch size | 32 |
| Loss | MSE |
| Optimiser | Adam, lr = 0.001 |
| Epochs | 100 |

## Results

Training loss falls steadily; validation loss plateaus around epoch 40 and then drifts upward
while training loss keeps dropping — the model starts overfitting the 320 training sequences.

| Epoch | Training loss | Validation loss |
| --- | --- | --- |
| 1 | 0.992 | 0.919 |
| 5 | 0.726 | 0.539 |
| 97 | 0.287 | 0.449 |
| 100 | 0.208 | 0.567 |

**Exact-match accuracy on the validation set is 0/80.** This is the honest headline: treating
next-address prediction as scalar regression and asking for a byte-exact 32-bit address is an
extremely demanding target, and MSE in normalised space does not optimise for it. The model
learns the broad shape of the trace without ever landing on an exact address.

The usual fixes, none of which are implemented here, would be to predict the **delta** rather
than the absolute address (the notebook already computes deltas and builds delta-based feature
arrays, but trains on absolute addresses), or to reframe the problem as **classification over a
vocabulary** of frequently-seen deltas rather than regression.

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
jupyter notebook cache_prefetcher.ipynb
```

Run the cells top to bottom. The three CSVs are committed, so the notebook is reproducible as-is.
Training runs on CPU and takes a few minutes.

There is a `long_feat` flag in the feature-engineering section, off by default, that switches to
overlapping windows (stride 10) instead of disjoint ones — many more training sequences at the
cost of a much longer run.
