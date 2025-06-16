FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir stable-baselines3 genesis-world matplotlib

# Install rsl_rl from source
RUN git clone https://github.com/leggedrobotics/rsl_rl.git && \
    cd rsl_rl && git checkout v1.0.2 && pip install -e . --use-pep517 && \
    cd .. && rm -rf rsl_rl

# Copy repo
WORKDIR /home/teru/ws
COPY . /home/teru/ws
RUN pip install -e .

CMD ["bash"]
