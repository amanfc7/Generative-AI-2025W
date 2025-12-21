# Setup & Run

## 1. Install Ollama
Make sure Ollama is installed on your system.


## 2. Pull the model and start the server
```bash
brew install ollama (in case you are on mac and have not installed ollama yet) no idea 
ollama serve
ollama pull mistral:7b-instruct
```

## 4. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```



## Run python script
```bash
python main.py
```

## Start web client
```bash
cd web-client
python -m http.server 8000
```

## See the web client
http://localhost:8000

