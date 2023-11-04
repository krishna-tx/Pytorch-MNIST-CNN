# Using PyTorch to Recognize Handwritten Digits

In this project, I used PyTorch to create a replica of the LeNet-5 CNN model to solve the famous MNIST dataset for handwritten digits.

### [Watch the YouTube Tutorial that goes step by step through the Jupyter Notebook](https://www.youtube.com/watch?v=ijaT8HuCtIY)
### [Read the Medium Blog Tutorial to Follow the Jupyter Notebook](https://medium.com/@krishna.ramesh.tx/training-a-cnn-to-distinguish-between-mnist-digits-using-pytorch-620f06aa9ffa)

To use the Python files first run "train_model.py". This will download the MNIST dataset if not already present, train the model, and save the best model to be used for testing. Then run "test_model.py" and provide image url link to the image you want to test it on. We deliberately swap the grayscale since majority of images have a light background so make sure to choose an image with a preferably white background for best results.

## Resources I used to gain knowledge for this project:

1. PadhAI [Deep Learning](https://padhai.onefourthlabs.in/courses/dl-feb-2019)
2. [Convolutional Neural Networks (CNNs) explained](https://www.youtube.com/watch?v=YRhxdVk_sIs&t=113s)
