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
