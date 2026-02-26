# BERT模型LoRA微调与Python API方案

## 1. 方案概述

本方案基于LoRA（Low-Rank Adaptation）技术对BERT模型进行参数高效微调，用于意图识别和组件生成任务。使用Python构建微调训练环境，并部署为API服务供前端调用，实现简单需求本地处理，复杂需求调用API的混合架构。

## 2. 技术选型

| 技术/库 | 版本/类型 | 用途 |
|---------|-----------|------|
| BERT模型 | bert-base-chinese | 预训练语言模型 |
| LoRA | peft | 参数高效微调 |
| PyTorch | ^2.0 | 模型训练和推理 |
| Hugging Face Transformers | ^4.40 | 模型加载和处理 |
| FastAPI | ^0.110 | API服务部署 |
| Uvicorn | ^0.29 | ASGI服务器 |
| React | ^18.3.1 | 前端集成 |

## 3. LoRA微调核心配置

### 3.1 LoRA基本原理
- **低秩适配**：仅训练少量低秩矩阵参数，而非整个模型
- **参数效率**：微调参数仅占原模型的0.1%-1%
- **训练加速**：减少GPU内存占用，加速训练过程
- **模型融合**：训练后可将LoRA权重与原模型融合，不影响推理速度

### 3.2 核心配置参数

| 参数名称 | 推荐值 | 说明 |
|----------|--------|------|
| `lora_r` | 8 | LoRA低秩矩阵的秩，控制可训练参数数量 |
| `lora_alpha` | 16 | LoRA缩放因子，通常设置为2*lora_r |
| `lora_dropout` | 0.1 | 正则化dropout概率 |
| `target_modules` | `['query', 'key', 'value']` | 需要应用LoRA的模型层（注意力机制的QKV层） |
| `bias` | `'none'` | 是否训练偏置参数 |
| `task_type` | `'SEQ_CLS'` | 任务类型（序列分类） |

### 3.3 训练配置

| 参数名称 | 推荐值 | 说明 |
|----------|--------|------|
| `learning_rate` | 1e-5 | 学习率，LoRA微调通常使用较小的学习率 |
| `per_device_train_batch_size` | 16 | 每设备训练批次大小 |
| `num_train_epochs` | 5 | 训练轮次 |
| `weight_decay` | 0.01 | 权重衰减 |
| `warmup_ratio` | 0.1 | 学习率预热比例 |
| `gradient_accumulation_steps` | 2 | 梯度累积步数 |
| `optim` | `'adamw_torch'` | 优化器 |

## 4. 微调数据集准备

### 4.1 意图识别数据集

```json
{
  "train": [
    {"text": "创建一个提交按钮", "label": "COMPONENT_CREATE"},
    {"text": "配置表单的API调用", "label": "API_INTEGRATION"},
    {"text": "调整按钮样式", "label": "STYLE_ADJUSTMENT"},
    {"text": "设置输入框的占位符", "label": "COMPONENT_CONFIGURE"}
  ],
  "test": [
    {"text": "添加一个新的选择器组件", "label": "COMPONENT_CREATE"},
    {"text": "修改表格的数据绑定", "label": "DATA_BINDING"}
  ]
}
```

### 4.2 组件生成数据集

```json
{
  "train": [
    {
      "prompt": "创建一个红色的提交按钮",
      "completion": {
        "componentType": "Button",
        "props": {
          "label": "提交",
          "type": "primary",
          "style": { "backgroundColor": "red" }
        }
      }
    }
  ],
  "test": [
    {
      "prompt": "创建一个用户名输入框",
      "completion": {
        "componentType": "Input",
        "props": {
          "label": "用户名",
          "placeholder": "请输入用户名"
        }
      }
    }
  ]
}
```

## 5. 微调实现

### 5.1 环境搭建

```bash
# 安装依赖
pip install torch transformers datasets peft scikit-learn fastapi uvicorn
```

### 5.2 微调训练代码

```python
# fine_tune_bert_lora.py

import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 加载数据集
dataset = load_dataset("json", data_files={
    "train": "intent_dataset_train.json",
    "test": "intent_dataset_test.json"
})

# 2. 加载预训练模型和分词器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(dataset["train"].unique("label")),
    id2label={0: "COMPONENT_CREATE", 1: "API_INTEGRATION", 2: "STYLE_ADJUSTMENT", 3: "COMPONENT_CONFIGURE"},
    label2id={"COMPONENT_CREATE": 0, "API_INTEGRATION": 1, "STYLE_ADJUSTMENT": 2, "COMPONENT_CONFIGURE": 3}
)

# 3. 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],
    bias="none"
)

# 4. 应用LoRA到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. 定义评估指标
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# 7. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# 8. 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# 9. 保存模型
model.save_pretrained("./lora-bert-intent")
tokenizer.save_pretrained("./lora-bert-intent")
```

### 5.3 运行微调

```bash
# 运行微调脚本
python fine_tune_bert_lora.py
```

## 6. Python API服务

### 6.1 API服务实现

```python
# api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

app = FastAPI(title="BERT Intent Recognition API")

# 加载微调后的模型和分词器
model_name = "bert-base-chinese"
peft_model_id = "./lora-bert-intent"

# 加载基础模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

# 定义请求和响应模型
class IntentRequest(BaseModel):
    text: str

class IntentResponse(BaseModel):
    intent: str
    confidence: float
    component_config: dict = None

# 意图识别端点
@app.post("/recognize-intent", response_model=IntentResponse)
async def recognize_intent(request: IntentRequest):
    try:
        # 模型推理
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取预测结果
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # 映射标签
        intent_map = {
            0: "COMPONENT_CREATE",
            1: "API_INTEGRATION",
            2: "STYLE_ADJUSTMENT",
            3: "COMPONENT_CONFIGURE"
        }
        
        intent = intent_map[predicted_class]
        
        # 生成简单的组件配置
        component_config = generate_component_config(request.text, intent)
        
        return IntentResponse(
            intent=intent,
            confidence=round(confidence, 4),
            component_config=component_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 组件配置生成函数
def generate_component_config(text: str, intent: str) -> dict:
    # 简单的规则生成，可根据实际需求扩展
    if intent == "COMPONENT_CREATE":
        if "按钮" in text:
            return {
                "componentType": "Button",
                "props": {
                    "label": "提交",
                    "type": "primary"
                }
            }
        elif "输入框" in text:
            return {
                "componentType": "Input",
                "props": {
                    "placeholder": "请输入内容"
                }
            }
    return None

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "lora-bert-intent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.2 启动API服务

```bash
# 启动API服务
python api_server.py
```

### 6.3 API测试

```bash
# 使用curl测试API
curl -X POST "http://localhost:8000/recognize-intent" \
  -H "Content-Type: application/json" \
  -d '{"text": "创建一个提交按钮"}'
```

## 7. 前端集成

### 7.1 API调用封装

```javascript
// src/services/aiService.js

const API_BASE_URL = 'http://localhost:8000';

/**
 * 调用本地微调模型API
 * @param {string} text 用户输入文本
 * @returns {Promise<Object>} 模型响应
 */
export const callLocalModel = async (text) => {
    try {
        const response = await fetch(`${API_BASE_URL}/recognize-intent`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('本地模型调用失败:', error);
        throw error;
    }
};

/**
 * 混合调用逻辑
 * @param {string} text 用户输入文本
 * @returns {Promise<Object>} 生成结果
 */
export const generateComponent = async (text) => {
    // 简单判断，决定调用本地模型还是API大模型
    if (text.length < 50 && !text.includes('复杂') && !text.includes('高级')) {
        console.log('使用本地微调模型');
        return await callLocalModel(text);
    } else {
        console.log('使用API大模型');
        // 调用现有API大模型
        return await callAPIModel(text);
    }
};
```

### 7.2 组件集成

```javascript
// src/components/AIGeneratePanel.js

import React, { useState } from 'react';
import { generateComponent } from '../services/aiService';

const AIGeneratePanel = () => {
    const [inputValue, setInputValue] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    
    const handleGenerate = async () => {
        if (!inputValue.trim()) return;
        
        setLoading(true);
        setError(null);
        
        try {
            const response = await generateComponent(inputValue);
            setResult(response);
            
            // 将生成的组件添加到画布
            if (response.component_config) {
                addComponentToCanvas(response.component_config);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="ai-generate-panel">
            <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="输入您的需求，如：创建一个提交按钮"
            />
            <button onClick={handleGenerate} disabled={loading}>
                {loading ? '生成中...' : '生成组件'}
            </button>
            
            {result && (
                <div className="result">
                    <h3>生成结果</h3>
                    <p>意图：{result.intent}</p>
                    <p>置信度：{result.confidence.toFixed(4)}</p>
                    {result.component_config && (
                        <div className="component-config">
                            <h4>组件配置</h4>
                            <pre>{JSON.stringify(result.component_config, null, 2)}</pre>
                        </div>
                    )}
                </div>
            )}
            
            {error && <div className="error">{error}</div>}
        </div>
    );
};

export default AIGeneratePanel;
```

## 8. 核心配置总结

| 配置项 | 核心值 | 说明 |
|--------|--------|------|
| 微调技术 | LoRA | 参数高效微调，仅训练低秩矩阵 |
| 模型基础 | bert-base-chinese | 中文BERT预训练模型 |
| LoRA参数 | r=8, alpha=16, dropout=0.1 | 低秩矩阵维度和缩放因子 |
| 训练参数 | lr=1e-5, epochs=5, batch_size=16 | 学习率、轮次和批次大小 |
| 部署方式 | FastAPI + Uvicorn | Python API服务 |
| API端口 | 8000 | 本地API服务端口 |
| 调用策略 | 基于文本长度和复杂度 | 简单需求调用本地模型，复杂需求调用API |

## 9. 预期效果

- **模型大小**：LoRA权重仅几十MB，易于部署
- **训练效率**：训练时间缩短70%以上
- **推理速度**：与原模型相当，无明显延迟
- **准确率**：意图识别准确率可达90%以上
- **API响应**：本地API响应时间<500ms
- **成本降低**：减少70%以上的API调用成本

## 10. 后续优化方向

1. **扩展数据集**：收集更多用户交互数据，提高模型泛化能力
2. **多任务微调**：同时训练意图识别和组件生成任务
3. **模型蒸馏**：将大模型知识蒸馏到本地模型，进一步提高性能
4. **自动更新机制**：实现模型的自动更新和版本控制
5. **优化调用策略**：基于模型置信度动态调整调用策略

## 11. 优势总结

1. **参数高效**：LoRA微调仅训练少量参数，资源消耗低
2. **快速部署**：Python API服务部署简单，易于维护
3. **响应迅速**：本地API响应速度快，用户体验好
4. **成本降低**：减少API调用，降低运营成本
5. **灵活扩展**：支持模型迭代更新，适应业务需求变化
6. **混合架构**：结合本地模型和API大模型的优势

这个方案结合了LoRA微调技术和Python API服务，实现了高效的本地AI能力，同时保持了系统的灵活性和扩展性。