#!/bin/bash
#Setup a virtual screen buffer
#Its extremely important for OpenAI to have a pixel depth of 24 when rendering the environment (e.g. 16 doesnt work)
Xvfb :1 -screen 0 800x600x24 &
#Setup a VNC server
x11vnc -display :1 -usepw -forever &
#Set the DISPLAY environment variable to be that created by Xvfb
DISPLAY=:1
export DISPLAY
#Start a lightweight desktop environment
fluxbox -display :1&
#Start a terminal that will cause the container to stop when exited
bash