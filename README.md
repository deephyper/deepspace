# DeepSpace

A collection of search space for the DeepHyper package.

## Requirements

Graphviz.

## Quick Start

```python
from deepspace.tabular import DenseSkipCoFactory

def_kwargs = dict(input_shape=(54,), output_shape=(7,), num_layers=10, dropout=0.0)
create_search_space = lambda *x: DenseSkipCoFactory(*x).create_space()
```
