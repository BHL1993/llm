import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig


# 加载基础模型和分词器
model_name = "/Users/baihailong/PycharmProjects/train/local_model/deepseek_15"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 加载LoRA配置和微调后的模型
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
checkpoint_dir = "./stats_result/checkpoint-2"  # 微调后保存的检查点目录
model = PeftModel.from_pretrained(base_model, checkpoint_dir, config=lora_config)


# 如果您的模型包含自定义部分（如交叉注意力层），请确保重新构建这些部分，并加载相应的权重

class QwenWithCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size

        # 统计特征编码器
        self.stats_encoder = StatsEncoder(hidden_size=self.hidden_size)

        # 交叉注意力层（查询来自文本，键值来自统计特征）
        self.cross_attn = CrossAttentionLayer(self.hidden_size)

        # 回归预测头
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, input_ids, stats_features, attention_mask=None, labels=None):
        # 文本编码 [batch, seq_len, hidden]
        text_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_hidden = text_outputs.hidden_states[-1]

        # 统计特征编码 [batch, hidden]
        stats_hidden = self.stats_encoder(stats_features)

        # 交叉注意力融合
        attn_output = self.cross_attn(text_hidden,stats_hidden)  #[batch_size, seq_len, hidden_size]

        # 取[CLS]位置预测概率
        pooled = attn_output[:, 0, :]   #[batch_size, hidden_size]
        # 假设第一个token是特殊标记
        return self.reg_head(pooled)


class StatsEncoder(nn.Module):
    """将统计特征编码到与文本隐藏层相同维度"""
    def __init__(self, stats_dim=3, hidden_size=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stats_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size)
        )

    def forward(self, stats):
        return self.mlp(stats)  # [batch, hidden_size]


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_hidden, stats_hidden):
        # text_hidden: [batch, seq_len, hidden]
        # stats_hidden: [batch, hidden]

        # 扩展统计特征维度
        stats_hidden = stats_hidden.unsqueeze(1)  # [batch, 1, hidden]

        # 计算注意力
        Q = self.q_proj(text_hidden)
        K = self.k_proj(stats_hidden)
        V = self.v_proj(stats_hidden)

        attn_weights = torch.matmul(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V)  # [batch, seq_len, hidden]
        return self.out_proj(attn_output)


# ... [与训练时相同的模型定义] ...

# 假设QwenWithCrossAttn是您完整的模型结构，这里我们需要将LoRA适配器应用到这个模型上
full_model = QwenWithCrossAttn()
full_model.base_model = model
full_model.to(torch.device('cpu'))  # 根据实际情况选择设备

# 准备输入数据
texts = ["物流延迟严重..."]
stats = [[3, 0.15, 36]]  # 示例统计特征
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# 进行推理
full_model.eval()  # 设置为评估模式
with torch.no_grad():
    outputs = full_model(input_ids=inputs["input_ids"],
                         attention_mask=inputs["attention_mask"],
                         stats_features=torch.FloatTensor(stats))
print(outputs)  # 输出预测结果