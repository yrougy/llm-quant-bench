REPO="unsloth/gemma-4-26B-A4B-it-GGUF"
CTX="16384"
CHAT_TPLT='{"enable_thinking":false}'

start_ls()
{
	/data/llamacpp/llama.cpp/llama-server \
	        -m "$1" \
	        -c "$CTX" \
	        -fit off  \
	        -fa on \
	        -b 1024         -ub 1024 \
	        --cache-type-k q4_0 \
	        --cache-type-v q4_0 \
	        --port 8050 \
	        --host 0.0.0.0 \
	        --temp 1.0 \
	        --top-p 0.95 \
	        --top-k 64 \
	        --chat-template-kwargs "${CHAT_TPLT}" > ./llama-server.log 2>&1 &
	        #--cache-type-k q4_0 \
	       # --cache-type-v q4_0 \

}

MODELS=$( ./list_quants.py $REPO | egrep -v 'F16|F32' )
for i in $MODELS
do
	HTTP_DL="https://huggingface.co/${REPO}/resolve/main/$i"
	# we test if it's already been benched before downloading
	if ls /data/benches/${i}-*
	then
		echo "$i already benched"
		continue
	fi
	echo Downloading $i
	mkdir -p "${REPO}"
	wget -c "${HTTP_DL}" -O "${REPO}"/"$i"
	start_ls "$REPO"/"$i"
	sleep 10
	./run.sh "$i"
	pkill llama-server
	rm -f "$REPO"/"$i"
	sleep 5
done
