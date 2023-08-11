import chess

import numpy as np

from copy import deepcopy

def piece_value(piece):
    '''
    ADD
    '''
    
    if piece.piece_type == chess.PAWN:
        return 1
    
    elif piece.piece_type == chess.KNIGHT:
        return 3
    
    elif piece.piece_type == chess.BISHOP:
        return 3
    
    elif piece.piece_type == chess.ROOK:
        return 5
    
    elif piece.piece_type == chess.QUEEN:
        return 9
    
    elif piece.piece_type == chess.KING:
        return 100

def trade_score(target : int, board : chess.Board) -> int:
    '''
    ADD
    '''
    
    attacked_fields = [i for i, val in enumerate(board.attackers(chess.WHITE, target).tolist()) if val == True]+[i for i, val in enumerate(board.attackers(chess.BLACK, target).tolist()) if val == True]
    involved_pieces = [board.piece_map()[field] for field in list(set(attacked_fields).intersection(board.piece_map().keys()))] #only get fields with pieces on them
    
    ally = [piece for piece in involved_pieces if piece.color == board.turn]
    opponent = [piece for piece in involved_pieces if piece.color != board.turn]
    
    a = []
    for piece in ally:
        a += [piece_value(piece)]
    
    b = []
    for piece in opponent:
        b += [piece_value(piece)]
    
    current_occupant = piece_value(board.piece_map()[target])
    
    #we sort all values involved besides the piece taken initially according to value
    a = sorted(a)
    b = sorted(b)
    
    #we insert the current occupant in the opponent value list here (after sorting!) since it was taken first and there is no choice for black to not trade it
    b.insert(0,current_occupant) 
    
    #we collect a list with the trade scores for each move one by one, starting with the first value directly after a took the current piece on the attacked field
    trade_score = [b[0]] #first we gain the current occupant piece
    
    #then we loop over a
    for i, a_val in enumerate(a):
        trade_score += [trade_score[-1]-a_val] #we lose the value of the piece which last took from black
        
        #if black still has pieces to trade, we gain these (given us having enough pieces to trade, see len block below)
        if i+1 < len(b):
            trade_score += [trade_score[-1]+b[i+1]]
        
        else:
            #if we weren't done looping over a when black did not have any pieces left
            if i < len(a):
                del trade_score[-1] #we remove our last loss since black actually could not have claimed the piece back
            
            break
    
    #if white has less pieces overall, then we weren't able to take blacks last occupant piece after all
    if len(a) < len(b):
        del trade_score[-1]
    
    #now all that is left is to find the value at which either party would not trade further
    #white and black will only trade equally or with a gain
    #so this means for all uneven/white moves, we only do them if the next white move i+2 yields a higher or equal value
    #if no i+2 field exists, we do not take since white has less pieces than black and would just lose a piece for free
    #for even/black moves, we only do them if at i+2 our value is equal or lower (since viewing from white perspective)
    #if no i+2 field exists we do not take
    #since black has the last say in even indices, we move one index further if that yields a lower or equal value
    
    idx = 0
    for i, val in enumerate(trade_score):
        #white moves
        if i % 2 == 1:
            #we test if i+2 exists
            if i+2 < len(trade_score):
                #we test i+2 value
                if val <= trade_score[i+2]:
                    idx += 1
                    
                else:
                    break
            else:
                break
        
        #black moves
        elif i % 2 == 0:
            #we test if i+2 exists
            if i+2 < len(trade_score):
                #we test i+2 value
                if val >= trade_score[i+2]:
                    idx += 1
                    
                else:
                    break
                    
    #we finally make another move for black if it is advantageous for black and possible
    if idx % 2 == 0 and idx < len(trade_score):
        if trade_score[idx] > trade_score[idx+1]:
            idx += 1
    
    return np.array([trade_score[idx]])

def heuristic_eval(move : str, board : chess.Board):
    '''
    Evaluates a given move, based on a defined heuristic, for the current board.
    The heuristic follows from a decision tree for which the root nodes contain the value of a given move.
    The values of all legal moves should be derived from it and a probability distribution should be made based on each move's value.
    The decision tree can be found in the directory of figures, and comments also annotate it in here; it is based on efficiency of computation.
    '''
    
    value = np.zeros(1)
    
    move = chess.Move.from_uci(move)
    
    #if the dictionary of pieces contains a piece for the target field it is a capture
    if move.to_square in board.piece_map().keys():
        #check if the target piece is defended (if the square is attacked by opponent more than zero times)
        if sum(board.attackers(not board.turn, move.to_square).tolist()) > 0:
            #in case of a defense, calculate the best possible value gain by any trading chain you can force
            value += trade_score(move.to_square, board)
        
        else:
            #since the piece is undefended calculate the attacked piece value and use it as the value of the move (can range from pawn=1 to queen=9)
            value += piece_value(board.piece_map()[move.from_square])
        
    #we only handle trades and captures due to computational load
    #ignoring undefended pieces and defense counts is favorable since it still enables the model to learn from selfplay taking it's pieces in the next move    
        
    return np.add(value.clip(-10,10),10)