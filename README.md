# Americant-Sign-Language-Alphabet-Recognition
This Project uses two types of inputs, Image input and skeleton data extracted from leap motion controller.
REAL-TIME RECOGNITION RESULTS
<img width="947" alt="image" src="https://github.com/Jnur9/Americant-Sign-Language-Alphabet-Recognition/assets/77942097/711bda3a-79dc-4ed6-8e4b-e88037bc8df6">
# Methodology 
To gather the data , a leap motion controller is used. Three deep learning models were created to improve the accuracy of the recognition system. These models take three different inputs: image input, skeleton input, and a combination of image and skeleton data. The recognition system is built using a user-friendly graphical user interface (GUI) that accesses the webcam and processes the captured sign. Once the software detects the sign, it will convert the hand movements into spoken language 
![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/03184511-f555-4a82-aa61-6ecd737f4b81)

![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/275cbabe-c8b1-4d31-a2d9-1f33d030cb13) 
# System Software
1)	Single- Input Model (Vison Module)
The first deep learning model uses one input to one output structure. The structure is a CNN (AlexNet), the dataset is scaled to 224 × 224 from 600 × 600 then converted to Grayscale and Gaussian filter was applied to reduce noises in the images.

2)	Single-Input (Skeleton Module)
The model utilizes skeletal data (hand landmark) and its architecture is a multi-layer feed-forward neural network (MLFFNN). The model input consists of 21 × 3 shape, with each joint in the hand represented by X, Y, and Z axis points. Essentially, the model takes in 3D hand landmark data as input.
![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/e65f071d-9c9d-4346-bc04-584d678ae455)

3)	Multi-Input (Mixed Module)
This model uses 2 different data, the images collected for the LMC and the 3D joint data of the sekleton. The structure of this model is basically illustrated in the figures below:

![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/0560f9a5-04a3-416f-ace6-e92d41445845)

![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/4ceccfc6-910a-44de-9872-e2e7b095257f)
# Simulation

Confusion Matrix Model 1

![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/3bffc932-e58f-45e8-8b4c-b2b0aca3734a)

Confusion Matrix Model 2

![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/865161dc-50f9-4178-bd71-c1cf09a39a73)

Model 3 predicted results 

![image](https://github.com/Jnur9/American-Sign-Language-Alphabet-Recognition-Using-Leap-Motion-Controller/assets/77942097/dd178755-c4a0-4d73-ac53-22f0bcc812d7)




 

 

  	 	 	 	 	 
    	     	 	 	 	 
