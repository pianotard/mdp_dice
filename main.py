import mdp
import pandas as pd
from board import Board

depth = 3
breadth = 10

def spawn(board, dice, index, pip = 1):
    try:
        return board.spawn_dice(dice, index, pip)
    except AssertionError as e:
        print(e)
        return board

def remove(board, index):
    try:
        return board.remove_dice(index)
    except AssertionError as e:
        print(e)
        return board

def merge(board, src, dest, dice):
    try:
        return board.merge_dice(src, dest, dice)
    except AssertionError as e:
        print(e)
        return board
    
def process_command(comd, board):
    """
    One of the following:
        - (S)pawn (D)ice (I)ndex (P)ip
        - (R)emove (I)ndex
        - (M)erge (S)ource (D)estination (N)ew_dice
    :param comd: string delimited by spaces
    :return: True if command was properly processed
    """
    tokens = comd.split(' ')
    if len(tokens) == 0:
        print('Empty command received')
        return board, True
    if tokens[0].lower() == 'e':
        print('Exit received')
        return board, False
    if tokens[0].lower() == 's':
        if len(tokens) == 4:
            return spawn(board, tokens[1], int(tokens[2]) - 1, int(tokens[3])), True
        else:
            print(f'Wrong number of args for (S)pawn: {len(tokens)}')
            return board, True
    if tokens[0].lower() == 'r':
        if len(tokens) == 2:
            return remove(board, int(tokens[1]) - 1), True
        else:
            print(f'Wrong number of args for (R)emove: {len(tokens)}')
            return board, True
    if tokens[0].lower() == 'm':
        if len(tokens) == 4:
            return merge(board, int(tokens[1]) - 1, int(tokens[2]) - 1, tokens[3]), True
        else:
            print(f'Wrong number of args for (M)erge: {len(tokens)}')
            return board, True
    print('Unknown command received')
    return board, True

def mdp_params(board):
    s_a_r_dict = board.mdp_params(depth, breadth)
    state_strings = []
    actions = []
    trans_probs = []

    for state, action_dict in s_a_r_dict.items():
        state_strings.append(state)
        for action, result in action_dict.items():
            state_strings.append(result.state_str())
            actions.append(action)
            trans_probs.append([state, action, result.state_str(), 1 / Board.DECK_LIMIT])

    state_strings = list(set(state_strings))
    actions = list(set(actions))
    
    return state_strings, actions, trans_probs

def run():
    
    deck = ['c', 'j', 'o', 'g', 'm']
    board = Board.new_board(deck)
    print(f'Initialized empty Board with deck: {deck}')
    print(board)
    
    def reward_func(state, action, result_state):
        return Board.parse_state_str(result_state, deck).dps() - Board.parse_state_str(state, deck).dps()
    
    while True:
        
        prompt(deck)
                
        try:
            states, actions, trans_probs = mdp_params(board)
            model = mdp.MDP(states, actions, trans_probs, reward_func, 0.5)
            pi_star = model.pi_stars[board.state_str()]
            print(f'Optimal step with depth {depth} and breadth {breadth}: {pi_star}')
        except:
            print(f'No optimal step here')
        
        command = input('Next command:\n')
        board, cont = process_command(command, board)
        
        if not cont:
            print('Bye!')
            break
            
        print(board)

def prompt(deck):
    print('---- COMMANDS ----')
    print('(S)pawn dice index pip')
    print('(R)emove index')
    print('(M)erge src dest new_dice')
    print('(E)xit')
    print('---- DECK ----')
    print(deck)
        
if __name__ == '__main__':
    run()