![Alt text](https://github.com/Homagn/MOVILAN/blob/main/Movilan_img.JPG?raw=true "Title")

0. Get familiarized with concepts:
(paper here: https://arxiv.org/abs/2101.07891)
(explanation slide here: https://github.com/Homagn/MOVILAN/blob/main/MOVILAN_detailed_explanation.pptx)

1. First set up the docker in your system
    if you dont have nvidia-docker then follow instructions :
    https://github.com/Homagn/Dockerfiles/blob/main/Docker-knowhows/nvidia-docker-setup

2. Pull the necessary environment or build it 
    Build the docker file :
    (download the source code here from github)
    (navigate to the Dockerfile location in MOVILAN/)
    
    sudo nvidia-docker build -t homagni/vision_language:latest .

    OR

    Pull the prebuilt docker image like this:
    
    docker pull homagni/vision_language:latest

3. Download the necessary model weights and data
    go to the google drive folder->  https://drive.google.com/file/d/1Spz3o5wmYUIMyXsYl3tKYYTMapzkca1_/view?usp=sharing
    download the zip file, after that extract the contents to your source MOVILAN/ folder like this 
    
    alfred_model_1000_modification -> language_understanding/alfred_model_1000_modification
    
    data -> mapper/data
    
    nn_weights -> mapper/nn_weights
    
    unet_weights.pth -> cross_modal/unet_weights.pth
    
    prehash.npy -> cross_modal/prehash.npy
    
    descriptions.json -> cross_modal/data/descriptions.json
    


4. Run the docker instance

    (in a terminal in linux)
    
    xhost +

    (after this in a newline)
    
    (NOTE- replace /home/homagni/Desktop/MOVILAN/ with the location where you have downloaded the source code)

    sudo nvidia-docker run --rm -ti --mount type=bind,source=/home/homagni/Desktop/MOVILAN/,target=/ai2thor --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=1" homagni/vision_language

    (Now youll be inside the terminal of the docker instance)
    
    (run the test code)
    
    cd /ai2thor
    
    python3 main.py

    (it should open up an ai2thor instance and run an execution of our algorithm for an instruction in ALFRED dataset)




EXTRA NOTES:

In mapper/params.py you can change debug_viz= True or false depending on whether you want to see the internal map state of the robot

The code creates a lot of log outputs depicting the various stages of decision making to make a log you can try

python3 main.py > SomeFile.txt 



(ERRORS ?)
If the display is not opening up from the docker instance (if youre using linux azure VM and docker from inside it)
https://github.com/mviereck/x11docker/issues/186
and
(probably the last instruction of this)
https://github.com/stas-pavlov/azure-glx-rendering

https://unix.stackexchange.com/questions/403424/x11-forwarding-from-a-docker-container-in-remote-server


using the --privileged flag as in here

(https://answers.ros.org/question/301056/ros2-rviz-in-docker-container/)

is able to make gazebo work with display from docker in azure cloud
