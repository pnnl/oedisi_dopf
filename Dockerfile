FROM python:3.10.6-slim-bullseye
USER root
RUN apt-get update && apt-get install -y git ssh
RUN mkdir -p /root/.ssh

# Libraries specifically required for Mac machines
RUN apt update && apt install -y \
  libboost-dev \
  libboost-filesystem-dev \
  libboost-program-options-dev \
  libboost-test-dev \
  libzmq5-dev python3-dev \
  libopenblas-dev \
  build-essential swig cmake git

WORKDIR /simulation
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser"] 
