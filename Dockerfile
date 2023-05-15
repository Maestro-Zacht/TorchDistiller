FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="8.0+PTX"
ENV IABN_FORCE_CUDA=1

RUN mkdir /code
WORKDIR /code

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip --no-cache-dir
RUN pip install wheel --no-cache-dir
RUN pip install -r SemSeg-distill/requirements.txt --no-cache-dir

CMD [ "echo", "'PyTorch test image'" ]