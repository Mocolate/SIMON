#From latest tensorflow image
FROM tensorflow/tensorflow

#Install system packages
#-x11vnc is a tool to create a VNC server
#-fluxbox provides a lightweight graphical environment for our VNC session
#-xvfb is a tool to create virtual screens
RUN apt-get update \
#    && apt-get upgrade -y \
    && apt-get install -y git \
    x11vnc \
    fluxbox \
    python-dev \
    cmake \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    libav-tools \
    xorg-dev \
    python-opengl \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && easy_install pip

#Copy the /src directory
RUN mkdir -p /root/workspace/
COPY src/ /root/workspace/

#Setup a password file for the VNC server
#RUN mkdir /root/.vnc
#RUN x11vnc -storepasswd 1234 ~/.vnc/passwd

#Setup a virtual screen buffer, with a VNC server
COPY run.sh /root/
RUN chmod +x /root/run.sh

#Define the startup script as the entrypoint of this container
ENTRYPOINT ["/root/run.sh"]

#Expose port 5900 for VNC connection
#EXPOSE 5900

#Set working directory
WORKDIR /root/

#Done!