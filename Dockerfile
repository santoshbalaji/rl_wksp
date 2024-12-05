# Use an Ubuntu-based image or another base that supports MuJoCo
FROM ubuntu:22.04

# Install required dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Set up MuJoCo environment (adjust this based on your MuJoCo installation)
ENV MUJOCO_GL egl

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY session_2/policy_grad_wk.py /app/policy_grad_wk.py
WORKDIR /app

# Command to run the Python script
CMD ["python3", "policy_grad_wk.py"]
