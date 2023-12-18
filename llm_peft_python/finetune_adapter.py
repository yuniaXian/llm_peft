""" This is a draft codes to implement adapter PEFT finetuning method.
    
class Adapter(nn.Module):
    create a layer called adaper (we finetune llm only on parameters in adapters)
        
class CombinedModel:
    bind one layer with another layer

class ModifiedTrainer:
    method save_model to save our modified model
    method compute_loss

For a use case, in main()
    It attaches the self_attention.dense layer in chatglm model with adapter (layer)
    For other llm model, specify which layers we want to add adapters to and modify the model

"""


from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments,TrainerCallback
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
tokenizer = AutoTokenizer.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True)
@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        # ids: question + answer
        ids = feature["input_ids"]
        # seq_len: the length of question part
        seq_len = feature["seq_len"]
        # use -100 as a special token to indicate that the postion require no prediction
        # [-100] * (seq_len - 1):  use -100 to mask the question part
        # ids[(seq_len - 1) :]: answer part (require prediction)
        # [-100] * (longest - ids_l)  no prediction on padding positions
        #  What   day   is    today  ?  [Answer        ] [PAD, ... , ...      ]
        # [-100, -100, -100, -100, -100, 0, 0, 0, ... 0, -100, -100, ..., -100]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))




class Adapter(nn.Module):
    def __init__(self, in_features, mid_features):
        super(Adapter, self).__init__() # or nn.Module.__init__(self)
        self.w1 = nn.Linear(in_features, mid_features)
        self.w2 = nn.Linear(mid_features, in_features)
        self.act= nn.ReLU()
    def forward(self, x):
        y = self.w1(x)
        y=self.act(y)
        delta_x = self.w2(y)
        return delta_x + x # 0.1* delta_x + x

# Bind any layer with another layer)
# we use it to bind adapter with self_attention.dense layer in chatglm2
class CombinedModel(nn.Module):
    def __init__(self, submodel1, submodel2):
        super(CombinedModel, self).__init__()
        #Warning: use float32 for training
       
        self.submodel1 = submodel1.to(torch.float32)
        self.submodel2 = submodel2.to(torch.float32)
 
    def forward(self, x):
        x=x.to(torch.float32)
        y1 = self.submodel1(x)
        y2 = self.submodel2(y1)
        return y2.half()
    # adapter=Adapter(in_features=10,mid_features=8)
    # print (adapter)

import pickle
def get_trainable_para_num(model):
    lst = []
    for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")


def main():
    os.environ["WANDB_DISABLED"] = "true"   
    writer = SummaryWriter()
    training_args = TrainingArguments(output_dir="chatglm-6b-adapter",per_device_train_batch_size=10,remove_unused_columns=False,num_train_epochs=1,learning_rate=1e-5)
    dataset_path="/axp/rim/novanlp/dev/jxian3/projects/llm/data/wenlv_token"
    # init model
  
    # Load model: float 16 for inference. For training we need float 32
    model = AutoModel.from_pretrained("/axp/rim/novanlp/dev/jxian3/projects/llm/chatglm/chatglm2", trust_remote_code=True,device_map="auto").cuda() 

    # Freeze all params in pretrained model. Train the adapter only
    for name, param in model.named_parameters():
        param.requires_grad=False


    adapter_list={}
    mid_features = 4


    for i,s in enumerate(model.transformer.encoder.layers, mid_features = mid_features):
        m=s.self_attention.dense
        # input dimension of adapter
        in_features=int(m.in_features)
        # trials
        mid_features=mid_features
        adapter=Adapter(in_features=in_features,mid_features=mid_features)
        # model is loaded in cpu by default，put it in cuda
        combined=CombinedModel(m,adapter).cuda()
        # replace the original layer with combined layer (adaptor + )
        s.self_attention.dense=combined
        adapter_list[i]=adapter
        # GPU cache is limited, we just add adapter to the first layer
        break
    

 
    # for name, param in model.named_parameters():
    #     print (name,param.requires_grad)
     
    get_trainable_para_num(model)
    print (model)

    dataset = datasets.load_from_disk(dataset_path)
    print (dataset)
    print(f"\n{len(dataset)=}\n")
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    # use save_model to ModifiedTrainer
    # model.save_pretrained(training_args.output_dir)
    # Warning: default model loader is not able to load adpator augmented model due to different architecture
 
    trainer.train()
    writer.close()
     
    for i,adapter in adapter_list.items():
        torch.save(adapter, training_args.output_dir+"/"+str(i))
    input="Hello, 你好"
    response, history = model.chat(tokenizer, input, history=[],max_length=200)
    print(response,len(response))

if __name__ == "__main__":
    main()
