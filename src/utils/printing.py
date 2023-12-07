import sys
from prettytable import PrettyTable


def print_debug(msg: str = '', force_exit: bool = False):

    print(f'>>> {msg}')
    if force_exit:
        sys.exit()


def pretty_confusion_matrix(matrix):

    columns = matrix.columns
    number_rows = matrix.shape[0]
    table_header = ['predicted']
    [table_header.append(column) for column in columns]

    pcm = PrettyTable()
    pcm.field_names = table_header
    for row_index in range(1, number_rows+1):
        row = [f'GT {row_index}']
        [row.append(matrix.at[row_index, column_index]) for column_index in range(1, len(columns)+1)]
        pcm.add_row(row)

    return pcm
