FROM python:3.10.6-slim-bullseye
RUN apt-get update
RUN apt-get install -y git ssh
RUN mkdir OptimalPowerFlow
COPY  . ./OptimalPowerFlow
WORKDIR ./OptimalPowerFlow
RUN pip install -r requirements.txt
EXPOSE 5903/tcp
CMD ["python", "server.py"]
