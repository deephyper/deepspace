import tensorflow as tf
from deephyper.nas.space import AutoKSearchSpace, SpaceFactory
from deephyper.nas.space.node import ConstantNode, VariableNode
from deephyper.nas.space.op.merge import Concatenate
from deephyper.nas.space.op.op1d import Dense


class OneLayerFactory(SpaceFactory):
    def __init__(self, input_shape=(10,), output_shape=(1,), regression=True):
        super().__init__(input_shape, output_shape, regression=regression)

    def build(self, input_shape, output_shape):
        ss = AutoKSearchSpace(input_shape, output_shape, regression=self.regression)

        if type(input_shape) is list:
            vnodes = []
            for i in range(len(input_shape)):
                vn = self.gen_vnode()
                vnodes.append(vn)
                ss.connect(ss.input_nodes[i], vn)

            cn = ConstantNode()
            cn.set_op(Concatenate(ss, vnodes))

            vn = self.gen_vnode()
            ss.connect(cn, vn)

        else:
            vnode1 = self.gen_vnode()
            ss.connect(ss.input_nodes[0], vnode1)

        return ss

    def gen_vnode(self) -> VariableNode:
        vnode = VariableNode()
        for i in range(1, 11):
            vnode.add_op(Dense(i, tf.nn.relu))
        return vnode


if __name__ == "__main__":
    factory = OneLayerFactory()
    # factory.test()
    # factory.plot_model()
    factory.plot_space()
