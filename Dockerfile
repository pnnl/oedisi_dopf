FROM python:3.10.6-slim-bullseye
# ARG SCENARIO
RUN apt-get update && apt-get install -y git ssh
RUN mkdir -p /root/.ssh

RUN apt update && apt install -y \
  libboost-dev \
  libboost-filesystem-dev \
  libboost-program-options-dev \
  libboost-test-dev \
  libzmq5-dev python3-dev \
  build-essential swig cmake git

WORKDIR /simulation
COPY . .
RUN pip install -r requirements.txt
# RUN oedisi build --component-dict scenario/$SCENARIO/components.json --system scenario/$SCENARIO/system.json --target-directory build
# ENTRYPOINT ["oedisi", "run", "--runner", "build/system_runner.json"]