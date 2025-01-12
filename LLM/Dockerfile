# Use an official lightweight Linux distribution as the base image
FROM ubuntu:22.04

# Set the environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama (replace the URL with the appropriate one for your platform)
RUN curl -fsSL https://ollama.com/install.sh | bash

# Start Ollama server temporarily to download the model
RUN nohup ollama serve & \ 
    sleep 5 && \
    ollama pull llama3.2 && \
    ollama pull nomic-embed-text && \
    pkill ollama

# Expose the default port for Ollama server
EXPOSE 11434

# Start the Ollama server
CMD ["ollama", "serve"]
