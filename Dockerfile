FROM python:3.10.6-slim-bullseye
USER root
ARG ALGO
ARG MODEL
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
COPY scenarios/$ALGO/$MODEL scenarios/
COPY outputs/$ALGO/$MODEL outputs/
COPY builds/$ALGO/$MODEL builds/
COPY feeder_federate .
COPY $ALGO_federate .
COPY recorder_federate .
COPY sensor_federate .
COPY generate_$ALGO.py .
COPY workflow_$ALGO.ipynb .
RUN pip install -r requirements.txt
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser"] 
