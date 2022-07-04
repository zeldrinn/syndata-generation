FROM ubuntu:20.04

# Set DEBIAN_FRONTEND to prevent TZ prompt from python3-opencv install
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y vim python3.8 python3-pip python3-opencv

WORKDIR /srv/syndata-generation

COPY . .

RUN pip install -r requirements.txt
