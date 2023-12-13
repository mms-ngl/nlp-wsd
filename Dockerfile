FROM python:3.9-slim

WORKDIR /home/app

# install requirements

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy model

COPY model model

# copy code

COPY wsd wsd
ENV PYTHONPATH wsd

# standard cmd

CMD [ "python", "wsd/app.py" ]
