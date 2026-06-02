FROM python:3.10-bullseye

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
