# Simulator sidecar ("sim"): the MCP server with this task's timetable and token. Never reachable from the
# agent except through http://sim:8000 (MCP tools and /health; /episode_log needs the token).
FROM python:3.11-slim
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt
COPY railway/ /srv/railway/
COPY timetable.json token /srv/
WORKDIR /srv
EXPOSE 8000
CMD ["python", "-m", "railway.net.server", "--timetable", "/srv/timetable.json", "--token-file", "/srv/token", "--host", "0.0.0.0", "--port", "8000"]
