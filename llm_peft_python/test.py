
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("chatglm2", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm2", trust_remote_code=True).half().cuda()
model = model.eval()


#langchain
promot="Shi Li is a CS professor working in University at Buffalo. He is my advisor"
question="Who is Shi Li"
input=promot+"Answer the question based on context:"+question
print (input)
response, history = model.chat(tokenizer, input, history=[])
print(response)

# Chinese version
promot="栗师是布法罗大学的计算机教授，他是我的导师"
question="栗师是谁"
input=promot+"根据上文回答问题:"+question
print (input)
response, history = model.chat(tokenizer, input, history=[])
print(response)
