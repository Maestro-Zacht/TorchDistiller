FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir

RUN mkdir /code
WORKDIR /code

COPY . .

CMD [ "echo", "'PyTorch test image'" ]