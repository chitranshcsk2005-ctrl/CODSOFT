# Tic-Tac-Toe AI using Minimax Algorithm
# Task 2 - CODSOFT Artificial Intelligence Internship

import math

board = [' ' for _ in range(9)]

def print_board():
    print()
    for i in range(0, 9, 3):
        print(board[i], '|', board[i+1], '|', board[i+2])
        if i < 6:
            print('--+---+--')
    print()

def is_winner(player):
    win_combinations = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    return any(all(board[i] == player for i in combo) for combo in win_combinations)

def is_draw():
    return ' ' not in board

def minimax(is_maximizing):
    if is_winner('O'):
        return 1
    if is_winner('X'):
        return -1
    if is_draw():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(False)
                board[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(True)
                board[i] = ' '
                best_score = min(score, best_score)
        return best_score

def ai_move():
    best_score = -math.inf
    move = 0
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    board[move] = 'O'

def player_move():
    move = int(input("Enter your move (1-9): ")) - 1
    if board[move] == ' ':
        board[move] = 'X'
    else:
        print("Invalid move!")
        player_move()

def game():
    print("Welcome to Tic-Tac-Toe!")
    print("You are X | AI is O")
    print_board()

    while True:
        player_move()
        print_board()
        if is_winner('X'):
            print("You win!")
            break
        if is_draw():
            print("It's a draw!")
            break

        ai_move()
        print("AI played:")
        print_board()
        if is_winner('O'):
            print("AI wins!")
            break
        if is_draw():
            print("It's a draw!")
            break

game()
