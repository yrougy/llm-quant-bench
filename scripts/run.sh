#!/usr/bin/env bash

source /data/benches/lm-evaluation-harness/.venv/bin/activate

MODEL_TESTED="$1"
MODEL="Qwen/Qwen3.6-35B-A3B"

echo bench de $1

echo $MODEL_TESTED

human_eval_test_chat()
{
export HF_ALLOW_CODE_EVAL="1"
lm_eval --model local-chat-completions\
	--model_args "model=${MDOEL},base_url=http://localhost:8050/v1/chat/completions,api_key=EMPTY,tokenizer=${MODEL}"\
 	--tasks "humaneval"\
	--batch_size 1\
	--confirm_run_unsafe_code\
	--log_samples\
	--output_path "/data/benches/${MODEL_TESTED}-human-eval"
}


humanp_eval_test()
{
export HF_ALLOW_CODE_EVAL="1"
lm_eval --model local-completions\
	--model_args "model=${MODEL},base_url=http://localhost:8050/v1/completions,api_key=EMPTY,tokenizer=${MODEL}"\
 	--tasks "humaneval_plus"\
	--batch_size 1\
	--confirm_run_unsafe_code\
	--log_samples\
	--output_path "/data/benches/${MODEL_TESTED}-human-eval"
}

humaninstruct_eval_test()
{
export HF_ALLOW_CODE_EVAL="1"
lm_eval --model local-completions\
	--model_args "model=${MODEL},base_url=http://localhost:8050/v1/completions,api_key=EMPTY,tokenizer=${MODEL}"\
 	--tasks "humaneval_instruct"\
	--batch_size 1\
	--confirm_run_unsafe_code\
	--log_samples\
	--output_path "/data/benches/${MODEL_TESTED}-human-eval_instruct"
}

human_eval_test()
{
export HF_ALLOW_CODE_EVAL="1"
lm_eval --model local-completions\
	--model_args "model=${MODEL},base_url=http://localhost:8050/v1/completions,api_key=EMPTY,tokenizer=${MODEL}"\
 	--tasks "humaneval"\
	--batch_size 1\
	--confirm_run_unsafe_code\
	--log_samples\
	--output_path "/data/benches/${MODEL_TESTED}-human-eval"
}

arc_chat_test() {

lm_eval --model local-completions\
   	--model_args "model=${MODEL},base_url=http://localhost:8050/v1/completions,api_key=EMPTY,tokenizer=${MODEL}"\
   	--tasks arc_challenge_chat\
    	--batch_size 1\
  	--apply_chat_template\
	--log_samples\
        --output_path "/data/benches/${MODEL_TESTED}-arc-chat"
}

arc_test() {

lm_eval --model gguf\
    --model_args "base_url=http://localhost:8050"\
    --tasks "arc_challenge"\
    --num_fewshot 8\
    --batch_size 1\
	--log_samples\
    --output_path "/data/benches/${MODEL_TESTED}-arc"

}

ifeval_test() {

lm_eval --model local-chat-completions\
      --model_args "model=${MODEL},base_url=http://localhost:8050/v1/chat/completions,api_key=EMPTY,tokenizer=${MODEL}"\
      --tasks "ifeval"\
      --num_fewshot 0\
      --batch_size 1\
      --apply_chat_template \
	--log_samples\
      --output_path "/data/benches/${MODEL_TESTED}-ifeval"

}

gsm8k_test() {
lm_eval --model local-chat-completions\
      --model_args "model=${MODEL},base_url=http://localhost:8050/v1/chat/completions,api_key=EMPTY,tokenizer=${MODEL}"\
     --tasks "gsm8k"\
     --num_fewshot 8\
     --batch_size 1\
     --apply_chat_template\
	--log_samples\
      --output_path "/data/benches/${MODEL_TESTED}-gsm8k"
}

mmlu_test() {
lm_eval --model gguf\
    --model_args "base_url=http://localhost:8050"\
    --tasks "mmlu"\
    --num_fewshot 5\
    --batch_size 1\
	--log_samples\
    --output_path "/data/benches/${MODEL_TESTED}-mmlu"

}

#arc_test >> $MODEL_TESTED-arc-log
ifeval_test >> $MODEL_TESTED-ifeval-log
gsm8k_test >> $MODEL_TESTED-gsm8k-log
#mmlu_test >> $MODEL_TESTED-mmlu-log
arc_chat_test >> $MODEL_TESTED-arc-chat-log
human_eval_test_chat >> $MODEL_TESTED-human-eval-log
#human_eval_test >> $MODEL_TESTED-human-eval-log
#humanp_eval_test >> $MODEL_TESTED-human-eval-plus-log
