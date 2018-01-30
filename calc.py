import click
import numpy as np
import tensorflow as tf

import config


def truncate(hours, marks):
    """
    Set mark to 0 if there was not given any time
    """
    for i in range(4):
        if hours[i] == 0:
            marks[i] = 0


@click.command()
def run():
    """
    Use this utility to check how trained neural network works.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f'{config.model_path}.meta')
        saver.restore(sess, config.model_path)

        hours = list(map(lambda e: int(e), input("Write discipline hours distribution vector (e.g. 10 12 0 0 3 17): ").split()))
        hours = np.array(hours).reshape((1, len(hours)))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('Times:0')
        nn = graph.get_tensor_by_name('Output_layer/Tanh:0')

        marks = sess.run(nn, feed_dict={x: hours})[0]
        truncate(hours[0], marks)
        s = sum(marks)
        marks = ' '.join([str(round(v/s * 100 * config.multiplier) / 100) for v in marks])
        print()
        print(marks)


if __name__ == '__main__':
    run()
