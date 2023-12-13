#!/bin/bash

# initial check

if [ "$#" != 1 ]; then
    echo "$# parameters given. Only 1 expected. Use -h to view command format"
    exit 1
fi

if [ "$1" == "-h" ]; then
  echo "Usage: $(basename "$0") [file to evaluate upon]"
  exit 1
fi

test_path=$1

# delete old docker if exists
docker ps -q --filter "name=nlp-wsd" | grep -q . && docker stop nlp-wsd
docker ps -aq --filter "name=nlp-wsd" | grep -q . && docker rm nlp-wsd

# build docker file
docker build . -f Dockerfile -t nlp-wsd

# bring model up
docker run -d -p 12345:12345 --name nlp-wsd nlp-wsd

# perform evaluation
/usr/bin/env python wsd/evaluate.py "$test_path"

# stop container
docker stop nlp-wsd

# dump container logs
docker logs -t nlp-wsd > logs/server.stdout 2> logs/server.stderr

# remove container
docker rm nlp-wsd