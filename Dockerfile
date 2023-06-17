FROM nvcr.io/nvidia/nemo:23.03

COPY ./requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /var/web
WORKDIR /var/web

RUN git clone https://github.com/RENCI/sam-serve.git

WORKDIR /var/web/sam-serve

ENV PYTHONPATH=/var/web/sam-serve

ENTRYPOINT uvicorn

CMD ['src.server:app']
