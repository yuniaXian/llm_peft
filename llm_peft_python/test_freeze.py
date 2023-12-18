
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm-6b-freeze", trust_remote_code=True).cuda()
#model = AutoModel.from_pretrained("E:\code\chatglm\chatglm2", trust_remote_code=True).cuda()

# Chinese version
input="纽约有什么好玩的地方？"
response, history = model.chat(tokenizer, input, history=[],max_length=50)
print(response)

input="What to do in NYC"
response, history = model.chat(tokenizer, input, history=[],max_length=50)
print(response)

