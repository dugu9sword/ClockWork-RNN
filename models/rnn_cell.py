import tensorflow as tf
import numpy as np
from typing import NamedTuple


def def_weight(name, in_size, out_size):
    return tf.get_variable(shape=[in_size, out_size],
                           initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                       stddev=np.sqrt(2 / (in_size + out_size))),
                           name=name)


def def_bias(name, size):
    return tf.get_variable(shape=[size],
                           initializer=tf.constant_initializer(0.0),
                           name=name)


RNNParam = NamedTuple("RNNParam", [("w_h", object), ("w_i", object), ("b", object)])


def apply_param(param, x_t, state):
    return tf.matmul(state, param.w_h) + tf.matmul(x_t, param.w_i) + param.b


def apply_param_with_cw_mask(param, x_t, state, cw_mask):
    return tf.matmul(state, cw_mask * param.w_h) + tf.matmul(x_t, param.w_i) + param.b


class AbstractRNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

    def zero_state(self, batch_size):
        return tf.Variable(tf.zeros(shape=[batch_size, self.hidden_size]), trainable=False)

    def _def_param(self, name):
        return RNNParam(w_h=def_weight("{}_weight_hidden".format(name), self.hidden_size, self.hidden_size),
                        w_i=def_weight("{}_weight_input".format(name), self.input_size, self.hidden_size),
                        b=def_bias("{}_bias".format(name), self.hidden_size))


class ClockWorkUtil:
    @staticmethod
    def compute_update_mask(current_time, modules, avg_hidden_size):
        ret = []
        for m in modules:
            if current_time % m == 0:
                ret.extend([1 for _ in range(avg_hidden_size)])
            else:
                ret.extend([0 for _ in range(avg_hidden_size)])
        return tf.cast(np.asarray(ret), dtype=tf.float32)

    @staticmethod
    def block_upper_triangular_matrix(block_size, block_num):
        total_size = block_size * block_num
        ret = []
        for i in range(total_size):
            zero_size = int(i // block_size) * block_size
            ret.append([0] * zero_size + [1] * (total_size - zero_size))
        ret = np.array(ret).reshape(total_size, total_size)
        return tf.cast(ret, dtype=tf.float32)


class RNNCell(AbstractRNNCell):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__(input_size, hidden_size)
        self.param = self._def_param("param")

    def __call__(self, x_t, state):
        new_state = tf.tanh(apply_param(self.param, x_t, state))
        return new_state, new_state


class GRUCell(AbstractRNNCell):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__(input_size, hidden_size)
        self.z = self._def_param("z")
        self.r = self._def_param("r")
        self.new = self._def_param("new")

    def __call__(self, x_t, state):
        z_gate = tf.sigmoid(apply_param(self.z, x_t, state))
        r_gate = tf.sigmoid(apply_param(self.r, x_t, state))
        new_state = tf.tanh(apply_param(self.z, x_t, r_gate * state))
        new_state = (1 - z_gate) * state + z_gate * new_state
        return new_state, new_state


class ClockWorkGRUCell(AbstractRNNCell):
    def __init__(self, input_size, hidden_size, modules):
        super(ClockWorkGRUCell, self).__init__(input_size, hidden_size)
        self.z = self._def_param("z")
        self.r = self._def_param("r")
        self.new = self._def_param("new")
        self.modules = modules

    def __call__(self, x_t, state, current_time):
        assert self.hidden_size % len(self.modules) == 0
        avg_hidden_size = self.hidden_size // len(self.modules)
        weight_mask = ClockWorkUtil.block_upper_triangular_matrix(avg_hidden_size, len(self.modules))
        update_mask = ClockWorkUtil.compute_update_mask(current_time, self.modules, avg_hidden_size)

        z_gate = tf.sigmoid(apply_param_with_cw_mask(self.z, x_t, state, weight_mask))
        r_gate = tf.sigmoid(apply_param_with_cw_mask(self.r, x_t, state, weight_mask))
        new_state = tf.tanh(apply_param_with_cw_mask(self.z, x_t, r_gate * state, weight_mask))
        new_state = (1 - z_gate) * state + z_gate * new_state

        new_state = new_state * update_mask + state * (1 - update_mask)
        return new_state, new_state


StateTuple = NamedTuple("StateTuple", [("c", object), ("h", object)])


class LSTMCell(AbstractRNNCell):
    def __init__(self, input_size, hidden_size, peephole=False):
        super(LSTMCell, self).__init__(input_size, hidden_size)
        self.f = self._def_param("f")
        self.i = self._def_param("i")
        self.new = self._def_param("new")
        self.o = self._def_param("o")
        self.peephole = peephole

    def zero_state(self, batch_size):
        return StateTuple(c=tf.Variable(tf.zeros(shape=[batch_size, self.hidden_size]), trainable=False),
                          h=tf.Variable(tf.zeros(shape=[batch_size, self.hidden_size]), trainable=False))

    def __call__(self, x_t, state):
        if isinstance(state, StateTuple):
            f_gate = tf.sigmoid(apply_param(self.f, x_t, state.c if self.peephole else state.h))
            i_gate = tf.sigmoid(apply_param(self.i, x_t, state.c if self.peephole else state.h))
            o_gate = tf.sigmoid(apply_param(self.o, x_t, state.c if self.peephole else state.h))
            new_cell_state = tf.tanh(apply_param(self.new, x_t, state.c if self.peephole else state.h))
            new_cell_state = f_gate * state.c + i_gate * new_cell_state
            new_hidden_state = o_gate * new_cell_state
            return new_hidden_state, StateTuple(c=new_cell_state, h=new_hidden_state)
        else:
            raise Exception


class ClockWorkLSTMCell(AbstractRNNCell):
    def __init__(self, input_size, hidden_size, modules, peephole=False):
        super(ClockWorkLSTMCell, self).__init__(input_size, hidden_size)
        self.f = self._def_param("f")
        self.i = self._def_param("i")
        self.new = self._def_param("new")
        self.o = self._def_param("o")
        self.peephole = peephole
        self.modules = modules

    def zero_state(self, batch_size):
        return StateTuple(c=tf.Variable(tf.zeros(shape=[batch_size, self.hidden_size]), trainable=False),
                          h=tf.Variable(tf.zeros(shape=[batch_size, self.hidden_size]), trainable=False))

    def __call__(self, x_t, state, current_time):
        if isinstance(state, StateTuple):
            assert self.hidden_size % len(self.modules) == 0
            avg_hidden_size = self.hidden_size // len(self.modules)
            weight_mask = ClockWorkUtil.block_upper_triangular_matrix(avg_hidden_size, len(self.modules))
            update_mask = ClockWorkUtil.compute_update_mask(current_time, self.modules, avg_hidden_size)

            f_gate = tf.sigmoid(
                apply_param_with_cw_mask(self.f, x_t, state.c if self.peephole else state.h, weight_mask))
            i_gate = tf.sigmoid(
                apply_param_with_cw_mask(self.i, x_t, state.c if self.peephole else state.h, weight_mask))
            o_gate = tf.sigmoid(
                apply_param_with_cw_mask(self.o, x_t, state.c if self.peephole else state.h, weight_mask))
            new_cell_state = tf.tanh(
                apply_param_with_cw_mask(self.new, x_t, state.c if self.peephole else state.h, weight_mask))

            new_cell_state = f_gate * state.c + i_gate * new_cell_state
            new_hidden_state = o_gate * new_cell_state

            new_cell_state = update_mask * new_cell_state + (1 - update_mask) * state.c
            new_hidden_state = update_mask * new_hidden_state + (1 - update_mask) * state.h

            return new_hidden_state, StateTuple(c=new_cell_state, h=new_hidden_state)
        else:
            raise Exception


class ClockWorkRNNCell(AbstractRNNCell):
    def __init__(self, input_size, hidden_size, modules):
        super(ClockWorkRNNCell, self).__init__(input_size, hidden_size)
        self.modules = modules

        self.cw = self._def_param("cw")

    def __call__(self, x_t, state, current_time: int):
        assert self.hidden_size % len(self.modules) == 0
        avg_hidden_size = self.hidden_size // len(self.modules)
        weight_mask = ClockWorkUtil.block_upper_triangular_matrix(avg_hidden_size, len(self.modules))
        update_mask = ClockWorkUtil.compute_update_mask(current_time, self.modules, avg_hidden_size)
        new_state = tf.tanh(tf.matmul(state, weight_mask * self.cw.w_h) + tf.matmul(x_t, self.cw.w_i) + self.cw.b)
        new_state = new_state * update_mask + state * (1 - update_mask)
        return new_state, new_state


def static_rnn(cell, inputs, batch_size):
    state = cell.zero_state(batch_size)
    outputs = []
    for i in range(len(inputs)):
        if hasattr(cell, "modules"):  # duck type
            output, state = cell(inputs[i], state, i + 1)
        else:
            output, state = cell(inputs[i], state)
        outputs.append(output)
    return outputs, state


# def static_multi_layer_rnn(cells, inputs, batch_size):
#     layer_num = len(cells)
#     states = []
#     for cell in cells:
#         state = cell.zero_state(batch_size)
#         states.append(state)
#     outputs = []
#     for time_step in range(len(inputs)):
#         to_feed = inputs[time_step]
#         for layer in range(layer_num):
#             cell = cells[layer]
#             if hasattr(cell, "modules"):  # duck type
#                 output, state = cell(to_feed, states[layer], time_step + 1)
#             else:
#                 output, state = cell(to_feed, states[layer])
#             to_feed = output
#             states[layer] = state
#         outputs.append(output)
#     return outputs, state  # return last-time-top-layer state!
