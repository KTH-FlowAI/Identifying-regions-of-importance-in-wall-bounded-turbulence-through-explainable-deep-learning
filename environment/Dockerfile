# hash:sha256:3dfec34b3efd4081661c501d2a4d33e7cd22b658309b3c137a47859990b1ee94
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
        jupyterlab==3.3.2 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    h5py==3.1.0 \
    keras==2.6.0 \
    matplotlib==3.6.1 \
    numpy==1.23.4 \
    protobuf==3.20.0 \
    scikit-learn==1.1.3 \
    shap==0.41.0 \
    tensorflow==2.6.1

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
