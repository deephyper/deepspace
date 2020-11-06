import tensorflow as tf
from deephyper.nas.space import KSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.op1d import Dense, Identity


class SupervisedRegAutoEncoderFactory(SpaceFactory):
    def __init__(
        self,
        input_shape=(10,),
        output_shape=[(1), (100,)],
        units=[128, 64, 32, 16, 8, 16, 32, 64, 128],
        num_layers=5,
        **kwargs
    ):
        super().__init__(
            input_shape, output_shape, unit=units, num_layers=num_layers, **kwargs
        )

    def build(self, input_shape, output_shape):
        ss = KSearchSpace(input_shape, output_shape)

        inp = ss.input_nodes[0]

        # auto-encoder
        units = [128, 64, 32, 16, 8, 16, 32, 64, 128]
        prev_node = inp
        d = 1
        for i in range(len(units)):
            vnode = VariableNode()
            vnode.add_op(Identity())
            if d == 1 and units[i] < units[i + 1]:
                d = -1
                for u in range(min(2, units[i]), max(2, units[i]) + 1, 2):
                    vnode.add_op(Dense(u, tf.nn.relu))
                latente_space = vnode
            else:
                for u in range(
                    min(units[i], units[i + d]), max(units[i], units[i + d]) + 1, 2
                ):
                    vnode.add_op(Dense(u, tf.nn.relu))
            ss.connect(prev_node, vnode)
            prev_node = vnode

        out2 = ConstantNode(op=Dense(100, name="output_1"))
        ss.connect(prev_node, out2)

        # regressor
        prev_node = latente_space
        # prev_node = inp
        for _ in range(self.num_layers):
            vnode = VariableNode()
            for i in range(16, 129, 16):
                vnode.add_op(Dense(i, tf.nn.relu))

            ss.connect(prev_node, vnode)
            prev_node = vnode

        out1 = ConstantNode(op=Dense(1, name="output_0"))
        ss.connect(prev_node, out1)

        return ss


if __name__ == "__main__":
    factory = SupervisedRegAutoEncoderFactory()
    # factory.test()
    factory.plot_model()
    # factory.plot_space()

