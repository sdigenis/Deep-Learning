# Deep-Learning

A collection of neural-network and machine-learning projects, ranging from networks built from
scratch in C to notebook-based experiments with scikit-learn.

## Projects

| Project | Description | Stack |
| --- | --- | --- |
| [Simple-Dense-Network](Simple-Dense-Network) | A dense multi-layer perceptron and a K-means implementation, both written from scratch for a Computational Intelligence course project. Includes accompanying exercise notebooks. | C, Jupyter |
| [MLP-From-Scratch-C](MLP-From-Scratch-C) | An earlier work-in-progress MLP for the same course project: hand-written forward pass, backpropagation, and SGD over a non-linearly separable 4-class dataset. | C, Jupyter |
| [Cache_prefetcher](Cache_prefetcher) | Smart caching — predicting a program's next memory reference from a trace of addresses, using K-means clustering over address deltas and a learned predictor. | Python, scikit-learn |

## Getting started

Each project is self-contained in its own directory. The C projects build with `make`:

```bash
cd MLP-From-Scratch-C
make run
```

The notebook projects need the usual scientific Python stack:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## License

Released under the [Apache License 2.0](LICENSE).
