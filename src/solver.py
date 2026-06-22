import numpy as np


def validate_sudoku_matrix(matrix):
    """Validate basic matrix shape and value range."""
    matrix = np.array(matrix)

    if matrix.shape != (9, 9):
        return False, "The matrix must have shape 9x9."

    if not np.all((matrix >= 0) & (matrix <= 9)):
        return False, "The matrix can only contain values from 0 to 9."

    return True, "Valid matrix."


def is_valid_group(values):
    """Check repeated values in a row, column, or block, ignoring zeros."""
    values = [int(value) for value in values if int(value) != 0]
    return len(values) == len(set(values))


def validate_sudoku_rules(board):
    """Validate Sudoku row, column, and 3x3 block rules for an incomplete board."""
    board = np.array(board)

    for row in range(9):
        if not is_valid_group(board[row, :]):
            return False, f"Conflict in row {row + 1}"

    for col in range(9):
        if not is_valid_group(board[:, col]):
            return False, f"Conflict in column {col + 1}"

    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            block = board[block_row:block_row + 3, block_col:block_col + 3].ravel()
            if not is_valid_group(block):
                return False, (
                    f"Conflict in block "
                    f"({block_row // 3 + 1}, {block_col // 3 + 1})"
                )

    return True, "The matrix follows the initial Sudoku rules."


def find_empty_cell(board):
    """Find the next empty cell in the board."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None


def is_valid_move(board, number, position):
    """Check whether a number can be placed in a given position."""
    row, col = position

    if number in board[row, :]:
        return False

    if number in board[:, col]:
        return False

    block_row = (row // 3) * 3
    block_col = (col // 3) * 3
    block = board[block_row:block_row + 3, block_col:block_col + 3]

    if number in block:
        return False

    return True


def solve_sudoku(board):
    """Solve a Sudoku board in place using backtracking."""
    empty_cell = find_empty_cell(board)

    if empty_cell is None:
        return True

    row, col = empty_cell

    for number in range(1, 10):
        if is_valid_move(board, number, (row, col)):
            board[row][col] = number

            if solve_sudoku(board):
                return True

            board[row][col] = 0

    return False


def validate_complete_solution(board):
    """Validate that a completed Sudoku solution is correct."""
    board = np.array(board)
    expected_values = set(range(1, 10))

    for row in range(9):
        if set(board[row, :]) != expected_values:
            return False, f"Error in row {row + 1}"

    for col in range(9):
        if set(board[:, col]) != expected_values:
            return False, f"Error in column {col + 1}"

    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            block = board[block_row:block_row + 3, block_col:block_col + 3].ravel()
            if set(block) != expected_values:
                return False, (
                    f"Error in block "
                    f"({block_row // 3 + 1}, {block_col // 3 + 1})"
                )

    return True, "Valid solution."


def solve_board(board):
    """Validate and solve a Sudoku board without mutating the original input."""
    board = np.array(board, dtype=int)

    format_valid, format_message = validate_sudoku_matrix(board)
    if not format_valid:
        return None, False, format_message

    rules_valid, rules_message = validate_sudoku_rules(board)
    if not rules_valid:
        return None, False, rules_message

    solution = board.copy()
    solved = solve_sudoku(solution)

    if not solved:
        return None, False, "No solution found."

    solution_valid, solution_message = validate_complete_solution(solution)
    if not solution_valid:
        return solution, False, solution_message

    return solution, True, solution_message
