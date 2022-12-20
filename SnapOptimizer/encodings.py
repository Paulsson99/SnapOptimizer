import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import json


class Encoding(ABC):

    @abstractmethod
    def get_encoding(self, N: int) -> np.ndarray:
        """
        Return an encoding with a Hilbert space of size N
        """


class FockEncoding(Encoding):

    def __init__(self, encoding_schema) -> None:
        self.encoding_schema = encoding_schema

    def get_encoding(self, N: int) -> np.ndarray:
        encoding = np.zeros((len(self.encoding_schema), N), dtype=np.complex128)

        for qubit_state, fock_states in self.encoding_schema.items():
            logic_state = np.zeros(N, dtype=np.complex128)
            logic_state[fock_states] = 1.0
            logic_state /= np.linalg.norm(logic_state)

            qubit_state_number = int(qubit_state, 2)
            encoding[qubit_state_number, :] = logic_state
        
        return encoding


def read_encoding(encoding_file: Path) -> Encoding:
    """
    Read an encoding from a file
    """
    with open(encoding_file, 'r') as f:
        encoding = json.load(f)
    if encoding['schema'] == 'fock':
        return FockEncoding(encoding['encoding'])
    raise NotImplementedError(f"The encoding schema {encoding['schema']} has not been implemnted yet")
