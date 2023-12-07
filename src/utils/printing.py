import sys
from prettytable import PrettyTable


def print_debug(msg: str = '', force_exit: bool = False):

    print(f'>>> {msg}')
    if force_exit:
        sys.exit()


def print_confusion_matrix(assessment_logger, matrix):

    columns = matrix.columns
    number_rows = matrix.shape[0]
    predicted_header = ['predicted']
    [predicted_header.append(column) for column in columns]

    x = PrettyTable()
    x.field_names = predicted_header
    for row_index in range(1, number_rows+1):
        row = [f'GT {row_index}']
        [row.append(matrix.at[row_index, column_index]) for column_index in range(1, len(columns)+1)]
        x.add_row(row)

    assessment_logger.info(msg='confusion matrix')
    assessment_logger.info(msg=x)
