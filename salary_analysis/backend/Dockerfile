FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY app.py .

COPY ai/ ai/
COPY agent/ agent/
COPY fetchers/ fetchers/
COPY langchain_graph/ langchain_graph/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
