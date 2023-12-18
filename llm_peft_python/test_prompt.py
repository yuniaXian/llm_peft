
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True)

# finetuned model
model1 = AutoModel.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm-6b-prompt-bak", trust_remote_code=True).cuda()
print ("prefix-tuning",model1)
#model = AutoModel.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True).cuda()
# original model

model2 = AutoModel.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True).cuda()
print ("orginial model",model2)

# input="What to do in NYC"
# response, history = model.chat(tokenizer, input, history=[])
# print(response)

