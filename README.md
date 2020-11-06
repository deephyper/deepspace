# DeepSpace

A collection of search space for the DeepHyper package.

## Requirements

Graphviz.

## Quick Start

```python
def create_search_space(input_shape=(54,), output_shape=(7,), num_layers=10, dropout=0.0, **kwargs):
    kwargs.update({k: v for k, v in locals().items() if k != "kwargs"})
    return DenseSkipCoFactory(**kwargs).create_space()
```
