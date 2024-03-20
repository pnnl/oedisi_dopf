FROM python:3.10.6-slim-bullseye
# ARG SCENARIO
RUN apt-get update && apt-get install -y git ssh
RUN mkdir -p /root/.ssh

WORKDIR /simulation
COPY . .
RUN pip install -r requirements.txt
# RUN oedisi build --component-dict scenario/$SCENARIO/components.json --system scenario/$SCENARIO/system.json --target-directory build
# ENTRYPOINT ["oedisi", "run", "--runner", "build/system_runner.json"]
