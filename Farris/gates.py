import numpy as np
qtype = np.complex64





class QuantumGate:

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self._dagger = False    
    
    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement matrix method.")

    def dagger(self) -> None:
        self._dagger = True

    def is_dagger(self) ->bool:
        return self._dagger
    


class Identity(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        # Define the Hadamard matrix for a single qubit
        hadamard = np.array([[1, 0], [0, 1]], dtype=qtype)
        return hadamard if not self._dagger else np.matrix.getH(hadamard)

    def dagger(self) -> None:
        self._dagger = True

    def is_dagger(self) ->bool:
        return self._dagger
    


class Hadamard(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        # Define the Hadamard matrix for a single qubit
        hadamard = np.array([[1, 1], [1, -1]], dtype=qtype) / np.sqrt(2)
        return hadamard if not self._dagger else np.matrix.getH(hadamard)

    def dagger(self) -> None:
        self._dagger = True

    def is_dagger(self) ->bool:
        return self._dagger
    



class PauliX(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        paulix = np.array([[0, 1], [1, 0]], dtype=qtype)
        return paulix if not self._dagger else np.matrix.getH(paulix)

    def __str__(self) -> str:
        return "X"
    



class PauliY(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        pauliy = np.array([[0, -1j], [1j, 0]], dtype=qtype)
        return pauliy if not self._dagger else np.matrix.getH(pauliy)

    def __str__(self) -> str:
        return "Y"
    


class PauliZ(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        pauliz = np.array([[1, 0], [0, -1]], dtype=qtype)
        return pauliz if not self._dagger else np.matrix.getH(pauliz)

    def __str__(self) -> str:
        return "Z"



class RotateX(QuantumGate):
    def __init__(self, theta: qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatex = np.array([[np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
                            [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)]], dtype=qtype)
        return rotatex if not self._dagger else np.matrix.getH(rotatex)

    def __str__(self) -> str:
        return f"Rx({self.theta})"
    

class RotateY(QuantumGate):
    def __init__(self, theta: qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatey = np.array(
            [[np.cos(self.theta / 2), -np.sin(self.theta / 2)], [np.sin(self.theta / 2), np.cos(self.theta / 2)]],
            dtype=qtype)
        return rotatey if not self._dagger else np.matrix.getH(rotatey)

    def __str__(self) -> str:
        return f"Ry({self.theta})"
    

class RotateZ(QuantumGate):
    def __init__(self, theta: qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatez = np.array([[np.exp(-1j * self.theta / 2), 0], [0, np.exp(1j * self.theta / 2)]], dtype=qtype)
        return rotatez if not self._dagger else np.matrix.getH(rotatez)

    def __str__(self) -> str:
        return f"Rz({self.theta})"
    

class Phase(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        phase = np.array([[1, 0], [0, 1j]])
        return phase if not self._dagger else np.matrix.getH(phase)

    def __str__(self) -> str:
        return "S"


class TGate(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=qtype)
        return t if not self._dagger else np.matrix.getH(t)

    def __str__(self) -> str:
        return "T"



class CNOT(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=qtype)
        return cnot if not self._dagger else np.matrix.getH(cnot)

    def __str__(self) -> str:
        return "CNOT"
    


class SWAP(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=qtype)
        return swap if not self._dagger else np.matrix.getH(swap)

    def __str__(self) -> str:
        return "SWAP"




class CRz(QuantumGate):
    def __init__(self, k:int) -> None:
        super().__init__(num_qubits=2)
        self._k=k

    def matrix(self) -> np.ndarray:
        crz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j*2*np.pi/(2**self._k))]], dtype=qtype)
        return crz if not self._dagger else np.matrix.getH(crz)

    def __str__(self) -> str:
        return "CRz("+self._k+")"
    