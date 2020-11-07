# DeepSpace

A collection of search space for the DeepHyper package.

## Requirements

Graphviz.

## Quick Start

```python
from deepspace.tabular import DenseSkipCoFactory

def create_search_space(input_shape=(54,), output_shape=(7,), **kwargs)
    return DenseSkipCoFactory()(input_shape, output_shape, num_layers=10, dropout=0.0)
```
