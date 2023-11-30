import sys


def print_debug(msg: str = '', force_exit: bool = False):

    print(f'>>> {msg}')
    if force_exit:
        sys.exit()
