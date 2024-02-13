## appn ----> Dockerfile(set of instructns ) ----> Docker image-----> docker container

FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y

RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "app.py"]