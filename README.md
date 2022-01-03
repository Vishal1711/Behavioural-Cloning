# Behavioural Cloning: End to End Learning for Self-driving Cars

The goal of this project was to train a end-to-end deep learning model that would let a car drive itself around the track in a driving simulator.

## Project Structure

| File  | Description |
| ------------- | ------------- |
|  IMG  |    Training data collected on Track 1 using right, left and centre camera |
| Drive.py  | Flask & Socket.io to establish bi-directional client-server communication  |
| behavioural_cloning.ipynb |  Code without data augmentation |
| behavioural_cloning_Final.ipynb  | Final Code  |
| driving_log.csv  | Collected Data - 'steering', 'throttle', 'reverse', 'speed' |
| model.h5  | Saved model after training |


## Data Collection and Blancing:

The provided driving simulator had two different tracks. One of them was used for collecting training data, and the other one — never seen by the model — as a substitute for test set.

The driving simulator would save frames from three front-facing "cameras", recording data from the car's point of view; as well as various driving statistics like throttle, speed and steering angle. We are going to use camera data as model input and expect it to predict the steering angle in the [-1, 1] range.

I have collected a dataset by driving on both direction around track 1. Driving in both direction reduce bias in collected datatset.
