# DeepSpace

A collection of search space for the DeepHyper package.

## Requirements

Graphviz.

## Quick Start

Generate a neural architecture space for fully connected networks with residual connections:

```python
from deepspace.tabular import DenseSkipCoFactory

def create_search_space(input_shape=(54,), output_shape=(7,), **kwargs)
    return DenseSkipCoFactory()(input_shape, output_shape, num_layers=10, dropout=0.0)
```

Generate a neural architecture space for AutoEncoder guided by an estimator:

```python
from deepspace.tabular import SupervisedRegAutoEncoderFactory

factory = SupervisedRegAutoEncoderFactory()(
     input_shape=(100,), output_shape=[(100), (10,)]
)
```
