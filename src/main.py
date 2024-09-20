"""
Program capable of mastering the classic game of "rock, paper, scissors" using Markov chain model.
Python 3.11
"""

from enum import Enum
import random
import numpy as np


class States(Enum):
    """
    Enum class that represents possible outcomes of player's last move.
    Value of each state also represents its position in matrix (except EM).
    """
    EM = -1  # Empty move (for the sake of first move)
    VR = 0  # Victory with rock
    VP = 1  # Victory with paper
    VS = 2  # Victory with scissors
    LR = 3  # Loss with rock
    LP = 4  # Loss with paper
    LS = 5  # Loss with scissors
    TR = 6  # Tie with rock
    TP = 7  # Tie with paper
    TS = 8  # Tie with scissors


what_beats_what_dict: dict[int, int] = {0: 2, 1: 0, 2: 1}  # 0 for rock, 1 for paper, 2 for scissors
decrease_value: float = 0.01
increase_value: float = 0.1
upper_limit: float = 1 - increase_value + decrease_value
bottom_limit: float = 0 + decrease_value

"""
 Matrix that represents the probabilities of transitioning between 9 states.
     VR VP VS LR LP LS TR TP TS
VR [ _, _, _, _, _, _, _, _, _]
VP [ _, _, _, _, _, _, _, _, _]
VS [ _, _, _, _, _, _, _, _, _]
LR [ _, _, _, _, _, _, _, _, _]
LP [ _, _, _, _, _, _, _, _, _]
LS [ _, _, _, _, _, _, _, _, _]
TR [ _, _, _, _, _, _, _, _, _]
TP [ _, _, _, _, _, _, _, _, _]
TS [ _, _, _, _, _, _, _, _, _]
"""
transition_matrix = np.asarray([[0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
                                [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]],
                               dtype=np.float64)


def player_move() -> int:
    """
    Function to prompt the player to make a move.

    The player can input either '1' for rock, '2' for paper, or '3' for scissors.

    Returns:
        int: An integer representing the player's move:
            0 for rock, 1 for paper, and 2 for scissors.
    """
    while True:
        move = input('Choose your move: 1-rock 2-paper 3-scissors: ')
        match move:
            case '1':
                return 0
            case '2':
                return 1
            case '3':
                return 2
            case _:
                print('Wrong choice!')


def computer_move(last_move_outcome: States) -> int:
    """
    Function to determine the computer's move.

    This function utilizes a simple strategy based on transition matrices and randomness.
    If the last move outcome is undecided (EM - empty), the function returns a random move.
    Otherwise, it calculates the decision vector based on the transition matrix and returns
    the move corresponding to the highest probability in the decision vector.

    Args:
        last_move_outcome (States): An enumeration representing the outcome of the last move.

    Returns:
        int: An integer representing the computer's move:
             0 for rock, 1 for paper, and 2 for scissors.
    """
    if last_move_outcome == States.EM:
        return random.randint(0, 2)

    vector = transition_matrix[last_move_outcome.value]
    decision_vector = np.asarray([vector[0]+vector[3]+vector[6],
                                  vector[1]+vector[4]+vector[7],
                                  vector[2]+vector[5]+vector[8]])

    counter_plays = {2: 0, 1: 2, 0: 1}  # 0 for rock, 1 for paper, 2 for scissors
    return counter_plays[int(decision_vector.argmax())]


def decide_round_result(player_choice: int, computer_choice: int) -> int:
    """
    Function to determine the outcome of a round.

    Args:
        player_choice (int): An integer representing the player's move:
                             0 for rock, 1 for paper, and 2 for scissors.
        computer_choice (int): An integer representing the computer's move:
                               0 for rock, 1 for paper, and 2 for scissors.

    Returns:
        int: An integer representing the outcome of the round:
             0 if the round ends in a tie.
             1 if the player wins the round.
             -1 if the player loses the round.
    """
    if player_choice == computer_choice:
        return 0

    if what_beats_what_dict[player_choice] == computer_choice:
        return 1

    return -1


def update_probabilities(player_choice: int, round_result: int,
                         last_move_outcome: States) -> States:
    """
    Function to update transition probabilities based on the outcome of a round.

    If the last move outcome is not undecided (EM - empty), and if no probability
    falls below 0 or exceeds 1, the function updates probabilities by fixed amount:
        decrease_value = 0.01
        increase_value = 0.1
    Changing this values may affect the learning speed of the algorithm.

    Args:
        player_choice (int): An integer representing the player's move:
                             0 for rock, 1 for paper, and 2 for scissors.
        round_result (int): An integer representing the outcome of the round:
                            1 if the player wins, -1 if the player loses, and 0 if it's a tie.
        last_move_outcome (States): An enumeration representing the outcome of the last move.

    Returns:
        States: The updated state representing the outcome of a round.
    """
    current_state: States
    if round_result == 1:
        current_state = States(player_choice)
    elif round_result == -1:
        current_state = States(player_choice+3)
    else:
        current_state = States(player_choice+6)

    if last_move_outcome != States.EM:
        vector = transition_matrix[last_move_outcome.value]
        if vector[current_state.value] <= upper_limit and np.min(vector) >= bottom_limit:
            vector[0:] = vector[0:] - decrease_value
            vector[current_state.value] += increase_value

    return current_state


def game_handler() -> None:
    """
    Function to handle the gameplay of a rock-paper-scissors game.

    This function manages the flow of the game, including player and computer moves, round outcomes,
    updating transition probabilities, and keeping track of the game score.

    Returns:
        None
    """
    last_move_outcome: States = States.EM
    round_counter: int = 0
    score: int = 0
    moves_dict: dict[int, str] = {0: 'rock', 1: 'paper', 2: 'scissors'}
    outcomes_dict: dict[int, str] = {0: 'It\'s a tie!', 1: 'You won!', -1: 'You lost!'}

    while round_counter < 30 and 10 > score > -10:
        print(f'-----Round: {round_counter} Score: {score}-----')
        player_choice = player_move()
        computer_choice = computer_move(last_move_outcome)
        print(f'You: {moves_dict[player_choice]} vs Computer: {moves_dict[computer_choice]}')

        round_result = decide_round_result(player_choice, computer_choice)
        print(outcomes_dict[round_result])

        last_move_outcome = update_probabilities(player_choice, round_result, last_move_outcome)
        score += round_result
        round_counter += 1

    print('-----GAME OVER!-----')
    if score == 0:
        print('It\'s a tie!')
    elif score > 0:
        print('You won! Congratulations')
    else:
        print('You lost! Better luck next time!')


if __name__ == '__main__':
    print('Welcome to classic "ROCK-PAPER-SCISSORS" game')
    print('You are going to play 30 rounds vs computer')
    print('Score is now set to 0. Win: +1, Loss: -1, Tie: 0')
    print('If score hits 10 - you win, if -10 - you lose')
    game_handler()
