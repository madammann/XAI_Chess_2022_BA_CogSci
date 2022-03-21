import numpy as np
import chess

def piece_from_char(char):
    color = chess.WHITE
    if char.islower():
        color = chess.BLACK
    piece = {'p' : chess.PAWN, 'b' : chess.BISHOP, 'n' : chess.KNIGHT, 'r' : chess.ROOK, 'q' : chess.QUEEN, 'k' : chess.KING}[char.lower()]
    return chess.Piece(piece,color)

class State:
    def __init__(self, board_epd, t=8, previous_states=[]):
        self.board_state = np.zeros((8,8,14)).astype('bool')
        self.board_past_states = []
        self.board_rights = np.zeros((8,8,7)).astype('bool')
        
        
#         morph = board_epd.split(' ')
#         morph[0] = morph[0].split('/')
        #Implement epd --> array
#         for x in range(self.board_state.shape[0]):
#             for line in morph[0]:
#                 i = 0
#                 fm = 0
#                 self.board_state[x][i][fm]
                
    def return_tensor(self):
        stack = None
        if len(self.board_past_states) > 0:
            stack = np.dstack([self.board_past_states[i] for i in range(len(self.board_past_states))])
            stack = np.dstack([self.board_state,stack,self.board_rights])
        else:
            stack = np.dstack([self.board_state,self.board_rights])
        return tf.constant(stack)
        

class Action:
    def __init__(self,action_tensor,board_state):
        self.move_distribution = []
        # Implement policy head out --> action uci
#         for x in range(len(action_tensor)):
#             for y in range(len(action_tensor[x])):
#                 pass