import tensorflow as tf

from deephyper.nas.space import SpaceFactory
from deephyper.nas.space import AutoKSearchSpace
from deephyper.nas.space.node import VariableNode
from deephyper.nas.space.op.op1d import Dense, Identity


class FeedForwardFactory(SpaceFactory):
    """Simple search space for a feed-forward neural network. No skip-connection. Looking over the number of units per layer and the number of layers.
    """

    def build(
        self,
        input_shape,
        output_shape,
        regression=True,
        num_units=(1, 11),
        num_layers=10,
        **kwargs
    ):
        """
        Args:
            input_shape (tuple, optional): True shape of inputs (no batch size dimension). Defaults to (2,).
            output_shape (tuple, optional): True shape of outputs (no batch size dimension).. Defaults to (1,).
            num_layers (int, optional): Maximum number of layers to have. Defaults to 10.
            num_units (tuple, optional): Range of number of units such as range(start, end, step_size). Defaults to (1, 11).
            regression (bool, optional): A boolean defining if the model is a regressor or a classifier. Defaults to True.

        Returns:
            AutoKSearchSpace: A search space object based on tf.keras implementations.
        """
        ss = AutoKSearchSpace(input_shape, output_shape, regression=regression)

        prev_node = ss.input_nodes[0]

        for _ in range(num_layers):
            vnode = VariableNode()
            vnode.add_op(Identity())
            for i in range(*num_units):
                vnode.add_op(Dense(i, tf.nn.relu))

            ss.connect(prev_node, vnode)
            prev_node = vnode

        return ss


if __name__ == "__main__":
    shapes = dict(input_shape=(10,), output_shape=(1,))
    factory = FeedForwardFactory()
    factory.test(**shapes)
    # factory.plot_model(**shapes)
    # factory.plot_space(**shapes)
