import csv
import openpyxl
import pathlib
import click
import tensorflow as tf
import numpy as np
import tqdm

import config
from calc import truncate


def network(sess, hours):
    hours = list(map(lambda e: int(e), hours))
    hours = np.array(hours).reshape((1, len(hours)))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('Times:0')
    nn = graph.get_tensor_by_name('Output_layer/Tanh:0')

    marks = sess.run(nn, feed_dict={x: hours})[0]
    truncate(hours[0], marks)
    return marks


def get_hours(ws):
    ws = ws['Робоча програма']
    pat = "Всього годин модуля"
    for r in ws:
        if r[0].value == pat:
            ind = r[0].row
            yield [ws[f"G{ind}"].value, ws[f"H{ind}"].value, ws[f"I{ind}"].value, ws[f"J{ind}"].value,
                   ws[f"K{ind}"].value, ws[f"L{ind}"].value]


def save_marks(ws, marks, row):
    __mark_cols = ['F', 'G', 'H', 'I', 'J', 'K', 'L']
    for c, v in zip(__mark_cols, marks):
        ws[f"{c}{row}"].value = v


def process(file_name, sess, out_dir, name='default.xlsx'):
    __mark_rows = [7, 52, 97, 142]
    __mark_total_col = 'O'
    try:
        f = openpyxl.load_workbook(file_name, data_only=True)
        marks_ws = f['Узагальнена Інформація ']
        for v, mr in zip(get_hours(f), __mark_rows):
            total = float(marks_ws[f"{__mark_total_col}{mr}"].value)
            marks = network(sess, v)
            s = sum(marks)
            marks = [str(round(v/s * 100 * total) / 100) for v in marks]
            save_marks(marks_ws, marks, mr)
        f.save(f"{out_dir}/{name}")
        f.close()
    except KeyError:
        pass
    # print(f"Processed: {out_dir}/{name}")


def iter_through(dir_: pathlib.Path):
    """
    Function that goes recursively through ``dir_``. If ``dir_'s`` value is
    directory it goes through its files.

    :param dir_: ``pathlib.Path`` instance
    """
    for v in dir_.iterdir():
        if v.is_dir():
            yield from iter_through(v)
        else:
            yield v


@click.command()
@click.argument('in_')
@click.option('--out', default='processed', help='Specify output directory. Otherwise it will save into processed/ directory')
def run(in_, out):
    """
    Retrieve data from xlsx files that are laid in given directory.
    First argument is .xlsx files directory; second is the directory for file to output data in.

    It uses pool with 4 processes to analyze files; so be careful when doing ``KeyboardInterrupt``

    *******
    Example
    *******

    ``python retriever.py c:/container/ d:/data.csv``

    """
    try:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(f'{config.MODEL_PATH}.meta')
            saver.restore(sess, config.MODEL_PATH)
            for v in tqdm.tqdm(list(iter_through(pathlib.Path(in_)))):
                process(v.as_posix(), sess, pathlib.Path(out).as_posix(), v.name)
        print(f"Data processed")
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run()