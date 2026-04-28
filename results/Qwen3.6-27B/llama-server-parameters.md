
```
                -c 16384 \
                -fit off  \
                -fa on \
                --cache-type-k q4_0 \
                --cache-type-v q4_0 \
                -b 1024         -ub 1024 \
                --port 8050 \
                --host 0.0.0.0 \
                --temp 0.6 \
                --top-p 0.95 \
                --top-k 20 \
                --min-p 0.00 \
                --chat-template-kwargs '{"enable_thinking":false}'
```
