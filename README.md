# llm
My finetuning codes for llms (PEFT)

### What's New:
* Nov 2023: Data compresser: Use entropy to 'filter in' sentences with high entropy, retaining sentences which are more informative) -> langchain/compress.py
* Oct 2023: Auto labelling prompt
* Oct 2023: Implement adapter finetuning for Chatglm-6b-2, add to any linear layer -> finetune_adapter.py
* Sep 2023: finetuning scripts for llms


Codes are based on llama2 and pretrained model: https://huggingface.co/THUDM/chatglm2-6b (newest update: chatglm3)






