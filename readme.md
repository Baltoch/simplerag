# 💬 Simple RAG

This project is a simple chatbot that uses Ollama, Tesseract OCR and ChromaDB to generate contextualized answers from provided documents.
The full stack runs locally on your machine, so it may be slow when running on lighter hardware configurations.

### Prerequisites

To run this project, you need to have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your machine (follow the links for more information).

You can also install [Docker Desktop](https://docs.docker.com/desktop/) instead, which includes Docker and Docker Compose by default.

To check if Docker is running, open a terminal and run the following command :

```bash
> docker -v
```

If Docker is running, you should see the version number displayed.

### Getting Started

Clone the repository and navigate to the project directory :

```bash
> git clone https://github.com/Baltoch/simplerag.git
> cd simplerag
```

Edit the .env.sample file by saving it as .env (the values are pre-filled with the default values)

Then, launch the project :

```bash
> docker compose up -d
```

You can now enjoy the [web interface](http://localhost:8501)

### Stopping the Project

If you want to stop the project, run the following command :

```bash
> docker compose down
```
