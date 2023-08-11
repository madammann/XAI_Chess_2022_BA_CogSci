import os
import argparse

from evaluate.mcts_eval import do_statistics as mcts_eval
from evaluate.model_vs_stockfish import model_vs_stockfish
from evaluate.testdata_eval import testeval
from evaluate.graphs import draw_all_graphs

#argparse arguments
parser = argparse.ArgumentParser(description='empty')

#path arguments for test dataset and stockfish engine
parser.add_argument('--datapath', default='./ChessDataset')
parser.add_argument('--modelpath', default='./weights/')
parser.add_argument('--modelname', default='Supervised')
parser.add_argument('--stockfish', default='./stockfish/stockfish-windows-2022-x86-64-avx2.exe')

#possible evaluations to call
parser.add_argument('--testdata_eval', default='False')
parser.add_argument('--stockfish_eval', default='False')
parser.add_argument('--mcts_eval', default='False')
parser.add_argument('--graphs', default='False')

parser.add_argument('--shutdown', default='False')
parser.add_argument('--n', default='1')

args = parser.parse_args()

def evaluate_testdata(datapath : str):
    testeval(datapath)

def evaluate_stockfish(stockfish : str, model_name : str, n=1):
    if model_name == 'all':
        model_vs_stockfish('Supervised',stockfish, n=n)
        model_vs_stockfish('Montecarlo',stockfish, n=n)
        model_vs_stockfish('DDPG',stockfish, n=n)
        
    else:
        model_vs_stockfish(model_name, stockfish, n=n)

def evaluate_mcts(stockfish_path, n=1):
    mcts_eval(stockfish_path, n=n)

def draw_graphs():
    draw_all_graphs()

def main(args):
    '''
    ADD
    '''
    try:
        if args.testdata_eval == 'True':
            evaluate_testdata(args.datapath)

        if args.stockfish_eval == 'True':
            evaluate_stockfish(args.stockfish, args.modelname, n=int(args.n))

        if args.mcts_eval == 'True':
            evaluate_mcts(args.stockfish, n=int(args.n))

        if args.graphs == 'True':
            draw_graphs()
    
    except Exception as e:
        print(e)
    
    if args.shutdown.lower() == 'true':
        print('Scheduled a shutdown in 3 minutes which may be stopped with "shutdown /a" in the cmd prompt.')
        os.system("shutdown -s -t 180") # schedule a shutdown that can be stopped with cmd("shutdown /a") if necessary.

if __name__ == '__main__':
    main(args)
