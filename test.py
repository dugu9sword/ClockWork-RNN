from models.rnn_cell import *


def winning_rate(target, prediction) -> float:
    return sum(np.array(target) * np.array(prediction) > 0)[0] / len(target)


time_steps = 100
feature_size = 1
batch_size = 50
max_epoch = 30

hidden_size = 100


def gen_x_y():
    x = 0.2 * np.random.random(size=batch_size * time_steps) - 0.1
    x = x.reshape(batch_size, time_steps)
    y = np.sum(x, axis=1).reshape(batch_size, 1)
    x = x.reshape(batch_size, time_steps, 1)
    return x, y


def main():
    # tf.contrib.eager.enable_eager_execution()

    ph_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, feature_size])
    ph_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    split_x = list(map(lambda a: tf.squeeze(a, axis=1),
                       tf.split(ph_x, axis=1, num_or_size_splits=time_steps)))

    # _, state = static_rnn(inputs=split_x,
    #                       cell=ClockWorkRNNCell(input_size=feature_size, hidden_size=hidden_size, modules=[1, 2]),
    #                       batch_size=batch_size)

    _, state = static_rnn(inputs=split_x,
                          cell=LSTMCell(input_size=feature_size, hidden_size=hidden_size),
                          batch_size=batch_size)

    if isinstance(state, StateTuple):
        state = state.h
    state = tf.reshape(state, [-1, hidden_size])
    dense_h = tf.layers.dense(inputs=state,
                              units=hidden_size / 2,
                              activation=tf.nn.relu)
    pred = tf.layers.dense(inputs=dense_h,
                           units=1,
                           activation=None)
    loss = 1 / 2 * tf.reduce_mean(tf.losses.mean_squared_error(labels=ph_y,
                                                               predictions=pred))
    opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch_id = 0
        while True:
            if epoch_id > max_epoch:
                break
            epoch_id += 1
            """ Train """
            x, y = gen_x_y()
            _, _loss = sess.run([opt, loss], feed_dict={
                ph_x: x,
                ph_y: y
            })
            print(str(_loss))

            """ Validation """
            x, y = gen_x_y()
            _loss, _pred = sess.run([loss, pred], feed_dict={
                ph_x: x,
                ph_y: y
            })

            rate = winning_rate(y, _pred)

            print("epoch: {:2}, test: {:10f}".format(epoch_id, rate))


if __name__ == '__main__':
    main()
