python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-qwen3-8b --output-dir ./results/rag/qwen3_8b --budget 10
python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-qwen3-8b --output-dir ./results/rag/qwen3_8b --budget 50
python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-qwen3-8b --output-dir ./results/rag/qwen3_8b --budget 100

python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-llama3.1-8b --output-dir ./results/rag/llama31_8b --budget 10
python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-llama3.1-8b --output-dir ./results/rag/llama31_8b --budget 50
python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-llama3.1-8b --output-dir ./results/rag/llama31_8b --budget 100

python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-gemma2-9b --output-dir ./results/rag/gemma2_9b --budget 10
python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-gemma2-9b --output-dir ./results/rag/gemma2_9b --budget 50
python -m iswc.rag.run_eval --dataset webqsp --pipeline native --model ollama-gemma2-9b --output-dir ./results/rag/gemma2_9b --budget 100
