# Deep-Learning

A collection of neural-network and machine-learning projects, ranging from networks built from
scratch in C to notebook-based experiments with PyTorch and scikit-learn.

## Projects

| Project | Description | Stack |
| --- | --- | --- |
| [Simple-Dense-Network](Simple-Dense-Network) | A multi-layer perceptron and K-means, both written from scratch in C with hand-implemented backpropagation, gradient clipping, and a numerically stable softmax. Classifies 2-D points into four non-linearly separable classes. Includes two visualisation notebooks. | C, Jupyter |
| [Cache_prefetcher](Cache_prefetcher) | Smart caching — predicting a program's next memory reference from its address trace, using K-means clustering over the address space and a two-layer LSTM. | Python, PyTorch, scikit-learn |

## Getting started

Each project is self-contained in its own directory, with its own README covering the approach,
architecture, and results.

The C project builds with `make`:

```bash
cd Simple-Dense-Network
make
./mlp
```

The notebook project needs the usual scientific Python stack plus PyTorch:

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

## License

Released under the [Apache License 2.0](LICENSE).
