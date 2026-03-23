FROM python:3.11-slim
WORKDIR /app
ARG ARG_PORT=8000
ENV PORT=${ARG_PORT}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE ${ARG_PORT}
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]