import torch
from transformers import AutoTokenizer, AutoModel
from finetune_adapter import  Adapter,CombinedModel
# use original tokenizer
tokenizer = AutoTokenizer.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True)
# load pretrained model
model = AutoModel.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True).cuda()
for i,s in enumerate(model.transformer.encoder.layers):
        adapter=torch.load("chatglm-6b-adapter/{}".format(i)).cuda()
        combined=CombinedModel(s.self_attention.dense,adapter)
        s.self_attention.dense=combined
        break

print (model)

# input="Could you show me some travelling spots in NYC? "
input="纽约有什么好玩的地方"


response, history = model.chat(tokenizer, input, history=[],max_length=200)
print(response)

