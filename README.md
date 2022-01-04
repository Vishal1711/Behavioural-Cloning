# Behavioural Cloning: End to End Learning for Self-driving Cars

The goal of this project was to train a end-to-end deep learning model that would let a car drive itself around the track in a driving simulator.

## Project Structure

| File  | Description |
| ------------- | ------------- |
|  IMG  |    Training data collected on Track 1 using right, left and centre camera |
| `Drive.py`  | Flask & Socket.io to establish bi-directional client-server communication  |
| `behavioural_cloning.ipynb` |  Code without data augmentation |
| `behavioural_cloning_Final.ipynb  `| Final Code  |
| driving_log.csv  | Collected Data - 'steering', 'throttle', 'reverse', 'speed' |
| `model.h5`  | Saved model after training |


## Data Collection and Balancing:

The provided driving simulator had two different tracks. One of them was used for collecting training data, and the other one — never seen by the model — as a substitute for test set.

The driving simulator would save frames from three front-facing "cameras", recording data from the car's point of view; as well as various driving statistics like throttle, speed and steering angle. We are going to use camera data as model input and expect it to predict the steering angle in the [-1, 1] range.

I have collected a dataset by driving on both direction around track 1. Driving in both direction reduce bias in collected datatset.

To filter front bias in steering data limit put at 400.

```
num_bins =25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
print(bins)
center = (bins[:-1]+ bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_p_bin, samples_per_bin))
```


![image](https://user-images.githubusercontent.com/79803663/147972871-52742f59-f94e-486c-bd52-ec409f7a6467.png)  ![image](https://user-images.githubusercontent.com/79803663/147972962-d83770f2-74d6-4e38-ad2d-f1b1a0bf1240.png)

## Data Augmentation and Preprocessing
After six laps of driving data we ended up with **6825 samples**, which most likely wouldn't be enough for the model to generalise well. However, as many pointed out, there a couple of augmentation tricks that should let you extend the dataset significantly:

1. Zoom Image - 
![image](https://user-images.githubusercontent.com/79803663/147973663-274b372c-a22f-4944-a378-08da8774b653.png)


2. Panned Image - 

![image](https://user-images.githubusercontent.com/79803663/147973787-f5f37de5-75aa-4435-ba71-c0cb49392b1d.png)

3. Brightness Changed -
 
 ![image](https://user-images.githubusercontent.com/79803663/147973984-5bff2bad-d5a3-491a-93cc-2b209defe013.png)

4. Flipped Image-

![image](https://user-images.githubusercontent.com/79803663/147974017-5d8bc740-379e-4a6f-b4cb-8b086c1b49f0.png)

5. Augmented Image-

<img width="812" alt="image" src="https://user-images.githubusercontent.com/79803663/147974208-6aac00a6-345c-4cce-b215-f7f73e6c1f8d.png">

Image Conversion in YUV Format-

![image](https://user-images.githubusercontent.com/79803663/147974320-91b38070-4745-4e94-a806-a3b0cd66cc6c.png)


## Model

I started with the model described in [Nvidia paper](https://arxiv.org/abs/1604.07316) and kept simplifying and optimising it while making sure it performs well on both tracks.

<img width="247" alt="image" src="https://user-images.githubusercontent.com/79803663/147975361-93ec066e-f6d8-406f-b9f7-58f2c0e2e875.png">

This model can be very briefly encoded with Keras.

```
def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(66,200,3), activation='elu'))
  model.add(Convolution2D(36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
  model.add(Convolution2D(48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
  model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))
  optimizer = Adam(learning_rate=0.001)
  model.compile(loss='mse', optimizer=optimizer)
  return model
```

## Result

The model was trained using arrow keys in the simulator on track 1. Therefore, the performance is not smooth as expected. Still, the car manages to drive just fine on both tracks. This is what driving looks like on track 2 (previously unseen).

Track 1:


https://user-images.githubusercontent.com/79803663/147994324-e85d8eb6-2fc4-4665-ba5c-07bb950edc1c.mp4


Track 2:

https://user-images.githubusercontent.com/79803663/147976999-50fd4141-d1ca-42c6-84c0-d2c144684f9c.mp4



  
