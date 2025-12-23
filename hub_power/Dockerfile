FROM python:3.10.6-slim-bullseye
RUN apt-get update
RUN apt-get install -y git ssh
RUN mkdir PowerHub
COPY  . ./PowerHub
WORKDIR ./PowerHub
RUN pip install -r requirements.txt
EXPOSE 5901/tcp
CMD ["python", "server.py"]
