## Introduction   

#### Background
Image classification problem is the task of assigning an input image one label from a fixed set of categories. This is one of the core problems in Computer Vision that, despite its simplicity, has a large variety of practical applications.


####  Challenges:  
Traditional way: Feature Description and Detection. 
![](1.png)
Maybe good for some sample task, but the actual situation is far more complicated. 
![](2.png)

Therefore, instead of trying to specify what every one of the categories of interest look like directly in code, we're going to use machine learning, which is providing the computer with many examples of each class and then develop learning algorithms that look at these examples and learn about the visual appearance of each class. 


## Purpose
Our purpose is to: 

1. Compare normal algorithms we learnt in class with deep learning on image classification problem.
2. Find a fast and acurate method that could run on a common laptop or smartphone. 
3. Explore the machine learning framework by Google - TensorFlow. 

## Dataset
The Oxford-IIIT Pet Dataset: [link](http://www.robots.ox.ac.uk/~vgg/data/pets/)

There are 25 breeds of dog and 12 breeds of cat. Each breed has 200 images. 

We only used 10 cat breeds in our project. 
![](3.png)



2.2 Dee