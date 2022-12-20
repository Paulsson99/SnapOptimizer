from pathlib import Path


def package(*args) -> Path:
    return Path(__file__).parent.joinpath(*args)

def toplevel(*args) -> Path:
    return package().parent.joinpath(*args)

def defaults(*args) -> Path:
    return package('defaults', *args)

def data(*args) -> Path:
    return toplevel('data', *args)

def pulses(*args) -> Path:
    return toplevel('pulses', *args)
