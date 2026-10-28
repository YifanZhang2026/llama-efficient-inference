#!/bin/bash
# Separate venvs on the data disk (do not touch the image's torch 2.8):
#   venv-vllm   vllm 0.16.0 + modelscope, and the HF-format Llama-3.2-1B
#   venv-quant  llm-compressor (GPTQ / AWQ); kept apart because pip could not
#               resolve vllm + llmcompressor together (it kept backtracking vllm)
set -u
R=/root/autodl-tmp
M=https://mirrors.cloud.tencent.com/pypi/simple
[ -x $R/venv-vllm/bin/pip ] || python -m venv $R/venv-vllm
$R/venv-vllm/bin/pip install "vllm==0.16.0" modelscope -i $M > $R/venv_setup_1029.log 2>&1
$R/venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
$R/venv-vllm/bin/modelscope download --model LLM-Research/Llama-3.2-1B --local_dir $R/hf/Llama-3.2-1B >> $R/venv_setup_1029.log 2>&1
ls $R/hf/Llama-3.2-1B
[ -x $R/venv-quant/bin/pip ] || python -m venv $R/venv-quant
$R/venv-quant/bin/pip install llmcompressor -i $M > $R/venv_quant_1029.log 2>&1
$R/venv-quant/bin/python -c "import llmcompressor, torch, transformers; print('llmcompressor', llmcompressor.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
echo "SETUP DONE $(date +%T)"
