# XAI_Chess_2022_BA_CogSci
Repository for my Cognitive Science Bachelor Thesis from 2022.

## Abstract
Will be added soon.

## Motivation
Playing chess and strategy games generally is interesting not only for entertainment but also for training the mind.    
I believe in the value of strategy games can bring and want to analyse them further.  
Since AlphaZero proved to be a milestone in chess, I want to implement a model based on it to analyse it with the goal to use Explainable AI techniques to extract implicit knowledge of Chess which the model posesses.  
The ultimate goal is to apply these concepts in similar domains to extract information to help humans learn from AI to learn close to optimal policies for a given domain.  

## About
This repository will, when the project is finalized, contain:
 * A trained model based on AlphaZero for chess.
 * Required libraries for interacting with the model.
 * A Monte-Carlo Tree Search algorithm for training the model.
 * A script for training the model.
 * Scripts for evaluation model training and performance.
 * Scripts for visualization and playing against the model.
 * A library of techniques applied to the model to analyse decision patterns.

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

### Setting up Stockfish
For evaluting the model performance Stockfish is used.  
Stockfish has to be downloaded manually and a path needs to be provided for it to be executed in the script.  
The path is currently stored in './stockfish_path.txt', it can be manually changed to whereever Stockfish was manually installed.

Step 1 : Download Stockfish from [here](https://stockfishchess.org/download/) (note that the used version by me is AVX2 - Windows).  
Step 2 : Place Stockfish in a desired file path and write the correct path for the `.exe` file in `stockfish_path.txt`.

## Structure
To be added...

## Contact
Marlon Dammann <<mdammann@uni-osnabrueck.de>>

## References
To be added...
