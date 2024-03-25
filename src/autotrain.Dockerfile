FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

# Set the working directory
WORKDIR /workspace/

RUN mkdir /workspace/.local /workspace/.cache && chmod 777 -R /workspace
COPY --chmod=777 ./ ./

# Install the dependencies
RUN apt-get update && apt-get install git-lfs && git lfs install
RUN pip install --no-cache-dir --upgrade pip && pip install autotrain-advanced==0.6.79 --force-reinstall && pip install requests botocore torch torchvision torchaudio
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN autotrain setup

# Run Jupyter Notebook on container startup
ENTRYPOINT ["./entrypoint.sh"]