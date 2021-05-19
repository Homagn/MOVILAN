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
    download the zip file, after that extract the contents to your source MOVILAN/ folder as directed in the file -> to_download


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
