Run these two shell commands and report their full output verbatim, then stop:
1. curl -sv --max-time 10 http://sim:8000/health
2. curl -sv --max-time 10 -X POST -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d '{"jsonrpc":"2.0","id":1,"method":"ping"}' http://sim:8000/mcp
