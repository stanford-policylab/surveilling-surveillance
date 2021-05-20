from pathlib import Path, PosixPath


def _is_path(file_path):
        return isinstance(file_path, (str, PosixPath))

