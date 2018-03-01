# SIMON
A deep learning homework assignment

## To run the docker image:
1. Install Docker as specified on https://www.docker.com/
2. Clone this repo
3. Cd to project root directory.
4. Build the Docker image: 'docker build -t simon .'
5. :coffee:
6. Run the Docker image: 'docker run --rm -it --name simon -p 8888:8888 -p 6006:6006 -p 15000:5900 -v /absolute/path/to/src/folder:/root/workspace/ simon'
	- Replace ''/absolute/path/to/src/folder' with the absolute path to the /src folder in the project directory.
	- This command mounts the project's /src folder into the container's /root/workspace folder, allowing the modifying of scripts while the container is running.

## Assignment
I was unable to work with the provided dataset, the file 'simonNetwork.py' shows the code with which I tried to extract the necessary information from the TFRecords file. 

Since the assigment is designed to test my skills in designing a CNN (I assume), I created a CNN for another very similar dataset: Zalando Fashion MNIST (https://github.com/zalandoresearch/fashion-mnist). 

There are two files in this repo that work with the zalando dataset: 'ZalandoFashionMNIST.py' (from https://medium.com/tensorist/classifying-fashion-articles-using-tensorflow-fashion-mnist-f22e8a04728a) and 'zalandoNetwork.py', which is my own CNN implementation. The first implementation is not convolutional and reaches a performance of 88% accuracy on the test set and 93% accuracy on the training set. My implementation is convolutional and reaches a performance of 89% accuracy on the test set and 97% accuracy on the training set. 

## To run the code:
1. Download the Zalando Fashion dataset from https://github.com/zalandoresearch/fashion-mnist (4 files)
2. Store the files in the src/input/data folder.
3. Train the cnn by running 'python zalandoNetwork.py' in the Docker container (bash). 
