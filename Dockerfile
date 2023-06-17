FROM nvcr.io/nvidia/nemo:23.03

RUN mkdir /var/web
WORKDIR /var/web

RUN git clone https://github.com/RENCI/sam-serve.git

WORKDIR /var/web/sam-serve

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/var/web/sam-serve

ENTRYPOINT uvicorn

CMD ['src.server:app']
