FROM python:3.10.6-slim-bullseye
#USER root
RUN apt-get update && apt-get install -y git ssh

RUN mkdir -p /root/.ssh

WORKDIR /simulation

COPY scenario .
COPY feeder_federate .
COPY measuring_federate .
COPY estimator_federate .
COPY recorder_federate .

RUN mkdir -p outputs

COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["./run.sh"]