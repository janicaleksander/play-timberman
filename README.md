
![App Screenshot](ss/ss.png)


# Play Timberman

An AI agent that plays the Timberman game using real-time computer vision and deep reinforcement learning.  

## Project Architecture  

The project consists of two integrated components:  

### Observer Module  
- Real-time screen processing and analysis system  
- Captures game frames and detects key elements (player position, branches, obstacles)  
- Extracts structured data to feed the AI model  

### AI Agent  
- Reinforcement learning agent based on Deep Q-Network (DQN)  
- Processes extracted visual data to learn optimal gameplay strategies  
- Continuously trains to maximize in-game performance  

## Features

- Fast and efficient screen processing with OpenCV  
- Accurate extraction of relevant gameplay data from the screen  
- Custom environment built with OpenAI Gymnasium for training and testing  
- Hyperparameter optimization for improved model training 

## Warning

The application was tested on a **2800 x 1800** screen.  
Variables describing the positions of elements detected by OpenCV are fixed for this resolution.  
If you want to run the project on your own screen, update these variables in `observer.py`.  


# Installation
```bash
pip install -r requirements.txt
```

Running the AI model only
```bash
python3 play-timberman.py
```
Running the observer to see real-time data extraction
```bash
python3 observer.py
```

