# 💬 GenAI Chat

### Getting Started

First edit the .env file by filling in your API keys :

```
# Clé API OpenAI
OPENAI_API_KEY=<Your OpenAI key> <-- Edit here

# Clé API LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<Your EU LangSmith key> <-- Edit here
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
LANGCHAIN_PROJECT=lab8-chatbot
```

Optionally you can setup and start a Python venv :

```bash
> python -m venv ./venv
> ./venv/Scripts/activate
```

You need to install dependencies :

```bash
> pip install -r requirements.txt
```

You can then launch the script :

```bash
> streamlit run .\\main.py
```

You can now enjoy the [web interface](http://localhost:8501)
