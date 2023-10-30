# Overview 
The purpose of this exercise is to create an AI dedicated in bypassing an ennemi. From a 12x12 matrix, the AI will learn to bypass
a player. The player and the AI are randomly placed. The player doesn't move and the AI must find a good path to bypass 
him.

To solve this problem, I use a Deep Q-Learning AI based on the stable baseline 3 (https://github.com/DLR-RM/stable-baselines3). The AI would bypass a player with 
the most realistic behavior to arrive behind him. The program is developed in Python, I did it with the IDE PyCharm. 
The AI needs to be clever enough to be efficient on a new map it has never seen during the training. 

# Deliverable

In this repository you will find two files: 
 
- "script.py": contains the main class containing all the methods required to create my custom environment.
- "bypass.py": contains the main matrix and the use of the class defined in the other file to train the AI and load 
the model.

# Requirements


- gym==0.26.2
- numpy==1.26.1
- pygame==2.5.2
- stable_baseline3==2.1.0

Python 3.9 at least is required to run this project. 

# Installation 

Clone the repository: 
```bash
https://github.com/Elena1505/Bypassing.git
cd Bypassing
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

# Usage
```bash
python model.py
python bypass.py
```