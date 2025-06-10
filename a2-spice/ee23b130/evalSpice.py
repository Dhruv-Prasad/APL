# EE23B130, Dhruv Prasad: Assignemnt1

from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, TextIO


def increaseDimensions(A: List[List[float]]) -> None:
    """Makes an n x n list of lists n+1 x n+1 with new elements as 0"""
    old_n = len(A)
    for rows in A:  # New column
        rows.append(0)
    A.append([0] * (old_n + 1))  # New row


def checkMalformed(file: TextIO) -> None:
    """Checks if circuit is 'valid', i.e.
    Lines starting with .circuit and .end exist and .end comes after,
    V, I, and R have sufficient and valid node and component data"""
    all_lines = file.readlines()
    idx = 0
    circuit_idx = (
        -1
    )  # This will remain -1 if there is no line that starts with .circuit
    end_idx = -1  # This will remain -1 if there is no line that starts with .end
    err = False
    # Any IndexError means sufficient data is not provided in file
    try:
        for line in all_lines:
            words = line.split()
            if len(words) == 0:
                words.append(".")
            if circuit_idx == -1 and end_idx == -1:
                # Get circuit_idx
                if words[0] == ".circuit":
                    circuit_idx = idx
            elif circuit_idx != -1 and end_idx == -1:
                # Get end_idx
                if words[0] == ".end":
                    end_idx = idx
                elif words[0][0] == "V" or words[0][0] == "I":
                    # Makes sure VSource and ISource are dc and have float values
                    if words[3] != "dc":
                        err = True
                        break
                    try:
                        float(words[4])
                    except:
                        err = True
                        break
                elif words[0][0] == "R":
                    # Makes sure resistance value is a positive float
                    try:
                        float(words[3])
                        if float(words[3]) <= 0:
                            err = True
                            break
                    except:
                        err = True
                        break
            elif circuit_idx != -1 and end_idx != -1:
                break
            idx += 1
    except IndexError:
        raise ValueError("Malformed circuit file")
    if circuit_idx == -1 or end_idx < circuit_idx or err == True:
        raise ValueError("Malformed circuit file")
    file.seek(0)


def handleNewNodes(
    unknowns: Dict[str, int],
    admittance_matrix: List[List[float]],
    constants_matrix: List[float],
    no_of_unknowns: int,
    words: List[str],
) -> int:
    """Checks if current component is connected to any new node and adds it to the unknowns Dict,
    Maintains and returns no_of_unknowns"""
    if words[1] not in unknowns:
        unknowns[words[1]] = no_of_unknowns
        no_of_unknowns += 1
        increaseDimensions(admittance_matrix)
        constants_matrix.append(0)
    if words[2] not in unknowns:
        unknowns[words[2]] = no_of_unknowns
        no_of_unknowns += 1
        increaseDimensions(admittance_matrix)
        constants_matrix.append(0)
    return no_of_unknowns


def handleResistance(
    unknowns: Dict[str, int], admittance_matrix: List[List[float]], words: List[str]
) -> None:
    """Updates admittance_matrix according to resistance"""
    admittance_matrix[unknowns[words[1]]][unknowns[words[1]]] += 1 / float(words[3])
    admittance_matrix[unknowns[words[2]]][unknowns[words[2]]] += 1 / float(words[3])
    admittance_matrix[unknowns[words[1]]][unknowns[words[2]]] -= 1 / float(words[3])
    admittance_matrix[unknowns[words[2]]][unknowns[words[1]]] -= 1 / float(words[3])


def handleISource(
    unknowns: Dict[str, int], constants_matrix: List[float], words: List[str]
) -> None:
    """Updates constants_matrix according to Isource"""
    constants_matrix[unknowns[words[1]]] -= float(words[4])
    constants_matrix[unknowns[words[2]]] += float(words[4])


def handleVSource(
    unknowns: Dict[str, int],
    admittance_matrix: List[List[float]],
    constants_matrix: List[float],
    no_of_unknowns: int,
    words: List[str],
) -> int:
    """Adds current through Vsource as unknown,
    Updates admittance_matrix and constants_matrix according to Vsource,
    Maintains and returns no_of_unknowns"""
    unknowns[words[0]] = no_of_unknowns
    no_of_unknowns += 1
    increaseDimensions(admittance_matrix)
    constants_matrix.append(0)
    admittance_matrix[unknowns[words[0]]][unknowns[words[1]]] += 1
    admittance_matrix[unknowns[words[1]]][unknowns[words[0]]] += 1
    admittance_matrix[unknowns[words[0]]][unknowns[words[2]]] -= 1
    admittance_matrix[unknowns[words[2]]][unknowns[words[0]]] -= 1
    constants_matrix[unknowns[words[0]]] += float(words[4])
    return no_of_unknowns


def handleGNDEquation(
    unknowns: Dict[str, int],
    admittance_matrix: List[List[float]],
    constants_matrix: List[float],
) -> None:
    """Changes GND KCL equation to V GND = 0"""
    for i in range(len(admittance_matrix[unknowns["GND"]])):
        admittance_matrix[unknowns["GND"]][i] = 0
    admittance_matrix[unknowns["GND"]][unknowns["GND"]] = 1
    constants_matrix[unknowns["GND"]] = 0


def evalSpice(filename: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Finds the unknown node voltages and currents through voltages for circuits containing R, dc V Sources and dc I Sources.
    Parameters :
    filename: Name of .ckt file to be solved
    Returns :
    V: Dictionary of Nodes (key) and their Voltages (value)
    I: Dictionary of Voltage sources (key) and Currents through them (value)
    """

    # Raise error if file of given name doesn't exist
    try:
        file = open(filename, "r")
    except:
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")

    # Check for malformed circuit
    checkMalformed(file)

    # Start reading of lines until .circuit is found
    line = file.readline()
    circuit_found = False
    end_found = False
    while not circuit_found:
        words = line.split()
        if len(words) > 0:
            if words[0] == ".circuit":
                circuit_found = True
        line = file.readline()

    # Initialize unknowns Dict, admittance_matrix, constants_matrix
    unknowns = defaultdict(int)
    no_of_unknowns = 0
    admittance_matrix = []
    constants_matrix = []

    # Go through lines of the file until .end is reached
    while not end_found:
        words = line.split()
        if end_found == False and len(words) > 0:
            if words[0] == ".end":
                end_found = True
            else:
                # Handle V, I, R components and raise error if others are found
                if words[0][0] == "R":
                    no_of_unknowns = handleNewNodes(
                        unknowns,
                        admittance_matrix,
                        constants_matrix,
                        no_of_unknowns,
                        words,
                    )
                    handleResistance(unknowns, admittance_matrix, words)
                elif words[0][0] == "I":
                    no_of_unknowns = handleNewNodes(
                        unknowns,
                        admittance_matrix,
                        constants_matrix,
                        no_of_unknowns,
                        words,
                    )
                    handleISource(unknowns, constants_matrix, words)
                elif words[0][0] == "V":
                    no_of_unknowns = handleNewNodes(
                        unknowns,
                        admittance_matrix,
                        constants_matrix,
                        no_of_unknowns,
                        words,
                    )
                    no_of_unknowns = handleVSource(
                        unknowns,
                        admittance_matrix,
                        constants_matrix,
                        no_of_unknowns,
                        words,
                    )
                else:
                    raise ValueError("Only V, I, R elements are permitted")
        line = file.readline()  # Read next line

    # Set V GND = 0
    handleGNDEquation(unknowns, admittance_matrix, constants_matrix)

    # Convert list of lists to numpy arrays
    np_admittance_matrix = np.array(admittance_matrix)
    np_constants_matrix = np.array(constants_matrix)

    # Solve the equations and raise error if it fails, i.e. has no or infinite solutions (current node, voltage loop)
    try:
        answer = np.linalg.solve(np_admittance_matrix, np_constants_matrix)
    except:
        raise ValueError("Circuit error: no solution")

    # Go through unknowns Dict and fill in V and I Dicts appropriately
    idx = 0
    V = {}
    I = {}
    for ele in unknowns:
        if ele[0] == "V":
            I[ele] = answer[idx]
        else:
            V[ele] = answer[idx]
        idx += 1
    return V, I
