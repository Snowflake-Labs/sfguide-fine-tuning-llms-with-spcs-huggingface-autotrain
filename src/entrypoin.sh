#!/bin/bash
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' > jupyter.out 2>jupyter.err &
echo "Fine-funing starting..."
nohup autotrain llm --train --project-name $PROJECT_NAME --model $MODEL_CARD --data-path stage/ --text_column text --use-peft --quantization int4 --lr 1e-4 --train-batch-size 4 --epochs 3 --trainer sft --token $HF_TOKEN --merge_adapter > autotrain.out 2>autotrain.err &
( tail -f -n0 autotrain.err & ) | grep -q "Saving target model..."
echo "Fine-funing complete. Merged model saved to stage."
tail -f /dev/null # Keeps container running