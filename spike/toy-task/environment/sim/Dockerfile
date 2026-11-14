FROM python:3.12-slim
RUN pip install --no-cache-dir mcp==2.3.0
COPY server.py token secret.txt /srv/
EXPOSE 8000
CMD ["python", "/srv/server.py", "--host", "0.0.0.0", "--port", "8000", "--token-file", "/srv/token"]
