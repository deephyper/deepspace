import collections

import tensorflow as tf

from deephyper.nas.space import AutoKSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.basic import Zero
from deephyper.nas.space.op.connect import Connect
from deephyper.nas.space.op.merge import AddByProjecting
from deephyper.nas.space.op.op1d import Dense, Identity, Dropout


class DenseSkipCoFactory(SpaceFactory):
    def build(
        self,
        input_shape,
        output_shape,
        regression=True,
        num_layers=10,
        dropout=0.0,
        **kwargs,
    ):
        ss = AutoKSearchSpace(input_shape, output_shape, regression=regression)
        source = prev_input = ss.input_nodes[0]

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        for _ in range(num_layers):
            vnode = VariableNode()
            self.add_dense_to_(vnode)

            ss.connect(prev_input, vnode)

            # * Cell output
            cell_output = vnode

            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(ss, [cell_output], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(ss, anchor))
                ss.connect(skipco, cmerge)

            prev_input = cmerge

            # ! for next iter
            anchor_points.append(prev_input)

        if dropout >= 0.0:
            dropout_node = ConstantNode(op=Dropout(rate=dropout))
            ss.connect(prev_input, dropout_node)

        return ss

    def add_dense_to_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case

        activations = [None, tf.nn.swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
        for units in range(16, 97, 16):
            for activation in activations:
                node.add_op(Dense(units=units, activation=activation))


if __name__ == "__main__":
    shapes = dict(input_shape=(10,), output_shape=(1,))
    factory = DenseSkipCoFactory()
    factory.test(**shapes)
    # factory.plot_model(**shapes)
    # factory.plot_space(**shapes)
