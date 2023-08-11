import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_all_losses():
    '''
    ADD
    '''
    
    supervised_train = list(pd.read_csv('./data/training_data_supervised.csv',index_col='Epoch')['Loss'])
    montecarlo_train = list(pd.read_csv('./data/training_data_montecarlo.csv',index_col='Episode')['Loss'])
    ddpg_train = list(pd.read_csv('./data/training_data_ddpg.csv',index_col='Epoch')['Loss'])
    
    supervised_test = list(pd.read_csv('./data/test_eval_supervised.csv',index_col='Epoch')['Loss'])
    montecarlo_test = list(pd.read_csv('./data/test_eval_montecarlo.csv',index_col='Epoch')['Loss'])
    ddpg_test = list(pd.read_csv('./data/test_eval_ddpg.csv',index_col='Epoch')['Loss'])
    
    return {'supervised' : (supervised_train, supervised_test), 'montecarlo' : (montecarlo_train, montecarlo_test), 'ddpg' : (ddpg_train, ddpg_test)}

def plot_test_losses(losses : dict):
    '''
    Draws a graph for the loss comparison for all models on the test data and saves it.
    
    :param model_name (str): ADD
    '''
    
    plt.style.use('bmh')
    
    fig = plt.figure()
    fig.set_size_inches(fig.get_size_inches()*2)
    
    plt.plot(losses['supervised'], label=f'Supervised model',c='blue')
    plt.plot(losses['montecarlo'], label=f'Montecarlo model',c='red')
    plt.plot(losses['ddpg'], label=f'DDPG model',c='green')
    
    plt.legend()
    
    plt.ylabel('Loss')
    plt.xlabel('Epoch/Episode')
    plt.title(f'Loss of all models on the test data')
    
    plt.savefig(f'./figures/test_losses_compared.png')
    
def plot_model_losses(loss : list, title, label, epoch=True, color='blue'):
    '''
    Draws a graph for the loss of a model over training episodes/epochs.
    
    :param model_name (str): ADD
    '''
    
    plt.style.use('bmh')
    
    fig = plt.figure()
    
    plt.plot(loss,label=label,c=color)
    
    plt.ylabel('Loss')
    xlabel = 'Epoch' if epoch else 'Episode'
    plt.xlabel(xlabel)
    plt.title(title)
    
    plt.savefig(f'./figures/{title.lower().replace(" ","")}.png')
    

# def times():
#     '''
#     ADD
#     '''
    
#     times_supervised = list(pd.read_csv('./data/training_data_supervised.csv',index_col='Epoch')['Runtime'])
#     times_montecarlo = list(pd.read_csv('./data/training_data_montecarlo.csv',index_col='Episode')['Runtime'])
#     times_ddpg = list(pd.read_csv('./data/training_data_ddpg.csv',index_col='Epoch')['Runtime'])
    
#     measurements = ('Average Epoch\ntime (min)', 'Total training time')
    
#     cmap = {'Supervised model' : 'blue', 'Monte Carlo model' : 'red', 'DDPG model' : 'green'}
    
#     data = {
#         'Supervised model' : np.array(np.mean([times_supervised]),np.sum([times_supervised])),
#         'Monte Carlo model' : np.array(np.mean([times_montecarlo]),np.sum([times_montecarlo])),
#         'DDPG model' : np.array(np.mean([times_ddpg]),np.sum([times_ddpg]))
#             }
    
#     width = 0.5
#     fig, ax = plt.subplots()
#     bottom = np.zeros(2)
    
#     for key, val in data.items():
#         p = ax.bar(measurements, val, width, label=key, bottom=bottom, color=cmap[key])
#         bottom += val

#     ax.set_title("Training time statistics between models")
#     ax.legend(loc="upper right")
    
#     plt.savefig(f'./figures/trainingtimes.png')
    
def stockfish_performance():
    df = pd.read_csv('./data/stockfish_vs.csv',index_col=None)
    
    all_models = df['Move Count']
    supervised = df.where(df['Model'] == 'Supervised').dropna()['Move Count']
    montecarlo = df.where(df['Model'] == 'Montecarlo').dropna()['Move Count']
    ddpg = df.where(df['Model'] == 'DDPG').dropna()['Move Count']
        
    models = ("All", "Supervised", "Montecarlo", "DDPG")
        
    per_elo_means = {
        'Max move count': (round(np.min(all_models),0), round(np.min(supervised),0), round(np.min(montecarlo),0), round(np.min(ddpg),0)),
        'Average move count': (round(np.mean(all_models),0), round(np.mean(supervised),0), round(np.mean(montecarlo),0), round(np.mean(ddpg),0)),
        'Min move count': (round(np.max(all_models),0), round(np.max(supervised),0), round(np.max(montecarlo),0), round(np.max(ddpg),0))
        
    }
    
    plt.set_cmap('viridis')
    plt.style.use('bmh')

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in per_elo_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average move count')
    ax.set_title('Average move counts per model')
    ax.set_xticks(x + width, models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 120)

    plt.set_cmap('viridis')
    
    plt.savefig(f'./figures/stockfish_results.png')

def get_hits(df):
    model = list(df['Best three'])
    stock = list(df['Stockfish best three'])
    
    hits = 0
    for i, model_triplet in enumerate(model):
        for move in model_triplet.split(','):
            if move in stock[i]:
                hits += 1
    
    return hits

def get_distance(df):
    model = list(df['MCTS value'])
    stock = list(df['Stockfish value'])
    
    mse = np.power(np.array(model) - np.array(stock), 2)
    
    return np.mean(mse)

def generate_comp_plot(per_model_stats, categories, title, y_label):
    names = list(categories)

    plt.set_cmap('viridis')
    plt.style.use('bmh')
    
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    axs[0].bar(names, list(per_model_stats.values())[0])
    axs[1].bar(names, list(per_model_stats.values())[1])
    
    axs[0].set_title('Time for 1000 iterations')
    axs[1].set_title('Best three hits')
    
    plt.savefig(f'./figures/{title.replace(" ","").lower()}.png')
    
def mcts_performance():
    df = pd.read_csv('./data/mcts_eval.csv',index_col=None)
    
    multithreaded = df.where(df['MCTS Mode'] == 'Multithreaded').dropna()
    vanilla = df.where(df['MCTS Mode'] == 'Vanilla').dropna()
    
    cnn = df.where(df['Simulation function'] == 'CNN').dropna()
    ran = df.where(df['Simulation function'] == 'RAN').dropna()
    
    per_version = {
        'Time': (multithreaded['Time'].mean(), vanilla['Time'].mean()),
        'Move Estimation': (get_hits(multithreaded), get_hits(vanilla))
    }
    
    per_simul = {
        'Time': (cnn['Time'].mean(), ran['Time'].mean()),
        'Move Estimation': (get_hits(cnn), get_hits(ran))
    }
        
    generate_comp_plot(per_version, ('Multi', 'Vanilla'), 'MCTS performance by simulation type', 'Percentage')
    generate_comp_plot(per_simul, ('CNN','RAN'), 'MCTS performance by algorithm', 'Percentage')
    
def draw_all_graphs():
    losses = get_all_losses()
    plot_test_losses({key : val[1] for key, val in losses.items()})
    
    plot_model_losses(losses['supervised'][0], 'Training loss of supervised model per epoch', 'Supervised Model', epoch=True, color='blue')
    plot_model_losses(losses['montecarlo'][0], 'Training loss of montecarlo model per episode', 'Montecarlo Model', epoch=False, color='red')
    plot_model_losses(losses['ddpg'][0], 'Training loss of DDPG model per epoch', 'DDPG Model', epoch=True, color='green')
    
    # times() #faulty
    
    stockfish_performance()
    
    mcts_performance()
        
#df.groupby('Model')['Move Count'].mean()