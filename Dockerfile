# ALFRED AI2THOR + Matterport 
# Requires nvidia gpu with driver 396.37 or higher


FROM nvidia/cudagl:10.1-devel-ubuntu18.04


# Install cudnn
#=============================================================
ENV CUDNN_VERSION 7.6.4.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*




#(ALFRED AI2THOR + Matterport)
#=============================================================
# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3-dev python3-pip git

RUN git clone https://github.com/askforalfred/alfred.git


RUN pip3 install opencv-python==4.1.0.25 torch==1.1.0 torchvision==0.3.0 numpy==1.13.3 pandas==0.24.1 networkx==2.2 h5py tqdm vocab revtok Pillow tensorboardX==1.8 ai2thor==2.1.0

RUN pip3 install git+https://github.com/Joeclinton1/google-images-download.git

#(some problems installing these unless upgrade pip)
RUN pip3 install --upgrade pip
RUN pip3 install img2vec_pytorch seqeval==0.0.12 transformers==2.7.0 pytorch-crf==0.7.2 dgl-cu101==0.4.3post2 webcolors gensim

#(make sure the ai2thor room data has been downloaded)
RUN apt-get install unzip
RUN mkdir -p $HOME/.ai2thor/releases/thor-201909061227-Linux64
RUN cd $HOME/.ai2thor/releases/thor-201909061227-Linux64 && wget http://s3-us-west-2.amazonaws.com/ai2-thor/builds/thor-201909061227-Linux64.zip && unzip thor-201909061227-Linux64.zip


#install latest cmake (for Matterport)
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

#(this one for exclusively matterport so for general exprt the path to whichever you like while container is run)
#ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build


#===============(ADD custom installs here while developing)====================
RUN pip3 install matplotlib
RUN pip3 install scikit-image
RUN pip3 install word2number









#(Execution instructions -Ai2Thor)
#=============================================================
#(say, the code youre developing is in /home/homagni/Desktop/ai2thor/ then mount it as follows)
#(remember to enable access to X11 forwarding)
# xhost +
#sudo nvidia-docker run --rm -ti --mount type=bind,source=/home/homagni/Desktop/MOVILAN/,target=/ai2thor --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=1" homagni/vision_language
#cd /ai2thor
#python3 main.py
#(it will show an execution trajectory)



#(Execution instructions -Matterport)
#=============================================================
#build part is done here, follow the remaining previous and post steps in matterport_setup instruction file
