FROM python:3.10.6-slim-bullseye
RUN apt-get update
RUN apt-get install -y git ssh
RUN mkdir Feeder
COPY  . ./Feeder
WORKDIR ./Feeder
RUN pip install -r requirements.txt
EXPOSE 5600/tcp
CMD ["python", "server.py"]
