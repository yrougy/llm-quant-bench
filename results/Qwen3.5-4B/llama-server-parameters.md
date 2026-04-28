
```
        -fit off  \
        -fa on \
        -b 1024         -ub 1024 \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --port 8050 \
        --host 0.0.0.0 \
        --temp 1.0 \
        --top-p 0.95 \
        --top-k 20 \
        --min-p 0.00 \
        --chat-template-kwargs '{"enable_thinking":false}'
```
