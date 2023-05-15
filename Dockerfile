FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="8.0+PTX"
ENV IABN_FORCE_CUDA=1
ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir /code
WORKDIR /code

COPY . .

RUN pip install --upgrade pip --no-cache-dir
RUN pip install wheel --no-cache-dir
RUN pip install -r /SemSeg-distill/requirements.txt --no-cache-dir

CMD [ "echo", "'PyTorch test image'" ]