import pandas as pd

def create_statistics_files():
    '''
    ADD
    '''
    
    training_df = pd.DataFrame(columns=['Epoch','Milestone','Device','Startdatetime','Enddatetime','Startepsilon','Loss','Path'])
    performance_df = pd.DataFrame(columns=['Epoch','Stockfish-1k','Stockfish-2k','Stockfish-3k','Custom-Eval-MCTS'])
    positional_df = pd.DataFrame(columns=['Epoch','ADD LATER'])
    opening_df = pd.DataFrame(columns=['Epoch','ADD LATER'])
    
    training_df.to_csv('./data/training_data.csv',index=False)
    performance_df.to_csv('./data/performance_data.csv',index=False)
    positional_df.to_csv('./data/positional_data.csv',index=False)
    opening_df.to_csv('./data/opening_data.csv',index=False)
    
def create_training_data_backup():
    training_df = pd.read_csv('./data/training_data.csv',index_col=None)
    datestring = datetime.now()
    datestring = f'{datestring.year}-{datestring.month}-{datestring.day}-{datestring.hour}-{datestring.minute}'
    training_df.to_csv(f'./backup/training_data_{datestring}.csv',index=False)
    
def add_training_data(epoch : int, milestone : int, loss : float, epsilon : float, starttime : str, endtime : str, path : str):
    '''
    ADD
    '''
    
    #create safety backup
    create_training_data_backup()
    
    #load .csv file and append data
    df = pd.read_csv('./data/training_data.csv',index_col=None)
    data = [epoch, milestone, 'ADD MANUALLY', starttime, endtime, epsilon, loss, path]
    appendix = pd.DataFrame([data],columns=['Epoch','Milestone','Device','Startdatetime','Enddatetime','Startepsilon','Loss','Path'])
    df = pd.concat([df,appendix])
    
    #save new dataframe
    df.to_csv('./data/training_data.csv',index=False)

def add_performance_data():
    '''
    ADD
    '''
    
    pass

def add_positional_data():
    '''
    ADD
    '''
    
    pass

def add_opening_data():
    '''
    ADD
    '''
    
    pass

def retrieve_training_progress():
    '''
    ADD
    '''
    
    try:
        pass
    
    except FileNotFoundError:
        create_statistics_files()
        return retrieve_training_progress() #recursively try again
        
    except Excection as e:
        raise