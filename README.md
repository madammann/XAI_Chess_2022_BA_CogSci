# XAI_Chess_2022_BA_CogSci
Repository for my Cognitive Science Bachelor Thesis from 2023.

## Abstract
In my bachelor thesis project I implement three models for playing chess using Deep Reinforcement Learning and Supervised Learning.
Additionally, Monte-Carlo Tree Search is implemented based on the multithreaded lock-free approach and the approach from the MCTS Solver paper.

These model architecture is based on the AlphaZero model.
The first model is the supervised model which was trained with a chess dataset from FICS.
The second model is the "monte-carlo" model which was trained with the same approach as in the AlphaZero paper.
The third model is the DDPG model, trained using Deep-Deterministic Policy Gradients.
These models are evaluated against Stockfish and the results are gathered and discussed.

## Repository structure
The repository consists of several folders, the three training scripts, an evaluation script and a script for preprocessing the dataset for the first model.

**Folders**
 - Data-Folder: Contains CSV files gathered during training and evaluation.
 - Evaluate-Folder: Contains scripts for different evaluation methods.
 - Figures-Folder: Contains the figures created from the graphs.py script which can be called using eval.py
 - Lib-Folder: Contains the ChessGame class and utility functions for interacting with the TensorFlow model.
 - Model-Folder: Contains MCTS implementation and the TensorFlow implementation of the AlphaZero model replica.
 - Weights-Folder: Contains the weights for loading in during training and evaluation, only the newest are included in the remote branch.
 
**Scripts**
 - dataset.py: Python script to preprocess the raw dataset from FICS and creating a dataset with training and test split.
 - eval.py: Python script for evaluating the models and MCTS implementation, can be called for gathering statistics with arguments.
 - training_ddpg.py: Python script for training the DDPG model, can be run at any time to continue training from last epoch.
 - training_montecarlo.py: Python script for training the "monte-carlo" model, can be run at any time to continue training.
 - training_supervised.py: Python script for training the supervised model, can be run at any time to continue but may run out of training data in this implementation after ~90 epochs.
 
**Other**
 - environment.yml: Anaconda environment file for creating the virtual environment with the right versions and libraries installed.
 - README.md: Contains this information about the repository.
 - MCTS Verify.ipynb: Contains some tests done for MCTS and allows for testing certain positions with MCTS and the model versions.

## Setup
In order to run the scripts in this repository a few installations and the environment setup must be done.

### Setting up the environment
The following setup guide is targeted at users who have Anaconda installed.  
If you have another setup with Visual Studio, Vim, etc. you will have to do the equivalent steps for your IDE.  
Firstly, all required libraries are listed in the `environment.yml` file.  
Additionally, if you intend to do the training too you must manually add cuda if you want the performance boost provided by it.

Step 1 : Download and install Anaconda, you can find the link [here](https://www.anaconda.com/products/distribution).  
Step 2 : Run the Anaconda prompt and navigate to the path this repository is cloned in.  
Step 3 : Create the base environment for this repository by running `conda env create -f environment.yml`.  
Step 4 : Activate the environment using `activate AlphaZeroChess2023`.  
Step 5 (optional) : Add additional libraries such as `cuda` or `jupyter lab` depending on your needs.

### Setting up the dataset
For training the supervised model and running test data evaluation on any model, a dataset is required.

Step 1 : [Download](https://www.ficsgames.org/download.html) all Standard "(average rating > 2000)" files from the time period of November 1998 - December 2022

*Option 1 - Standard Path*  
Step 2 : Create a folder named dataset inside this repository and move all files to it
Step 3 : Unzip all files and then delete everything in the folder which is not a ".PGN" file.
Step 4 : Run the dataset.py script to preprocess the dataset.

*Option 2 - Manual Path*  
Step 2 : Move all files to a desired location and later, when in the dataset.py script call or the training and testdata eval calls, provide a "--datapath" or "--data" argument with the path.
Step 3 : Unzip all files and then delete everything in the folder which is not a ".PGN" file.
Step 4 : Run the dataset.py script to preprocess the dataset.

### Setting up Stockfish
For evaluting the model performance Stockfish is used.  
Stockfish has to be downloaded manually and a path needs to be provided for it to be executed in the script.  
The path is currently stored in './stockfish_path.txt', it can be manually changed to whereever Stockfish was manually installed.

Step 1 : Download Stockfish from [here](https://stockfishchess.org/download/) (note that the used version by me is AVX2 - Windows).  

*Option 1 - Standard Path*  
Step 2 : Create a folder called stockfish in this repository at the root and unpack the zip folder which was downloaded in it.
Step 3 : Make sure the executable file is in the created folder and not somewhere else, it is expected at "./stockfish/stockfish-windows-2022-x86-64-avx2.exe".

*Option 2 - Manual Path*  
Step 2 :  Unzip the downloaded files anywhere and use a "--stockfish" or "--stockfish_path" argument when calling scripts which need Stockfish.

The Stockfish parameters can be found in "evaluate/stockfish_wrapper.py", during evaluation defaul values except for Elo were used, based on my hardware.

## Contact
Marlon Dammann <<mdammann@uni-osnabrueck.de>>

## References
M. Enzenberger and M. Müller, *"A lock-free multithreaded Monte-Carlo tree search algorithm"* in 
Advances in Computer Games, Berlin, Heidelberg: Springer Berlin Heidelberg, 2009, pp. 14-20.

Winands, M. H., et al. *"Monte-Carlo tree search solver."* Proceedings of the 6th International 
Conference on Computers and Games, CG 2008, Beijing, China, September 29-October 1, 2008. 
Springer Berlin Heidelberg, 2008, pp. 25-36.

Silver, David, et al. *"Mastering Chess and Shogi by Self-Play with a General Reinforcement 
Learning Algorithm."* arXiv:1712.01815 (2017).