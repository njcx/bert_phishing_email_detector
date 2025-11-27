import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
from tqdm import tqdm


class PhishingEmailDataset(Dataset):
    """钓鱼邮件数据集"""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 组合邮件特征：标题 + 正文 + 附件名
        title = str(row['title']) if pd.notna(row['title']) else ""
        body = str(row['body']) if pd.notna(row['body']) else ""
        attachment = str(row['attachment_name']) if pd.notna(row['attachment_name']) else ""

        # 构建输入文本：使用 [SEP] 分隔不同字段
        # 格式: [CLS] 标题 [SEP] 正文 [SEP] 附件 [SEP]
        if attachment and attachment != 'nan':
            text = f"{title} [SEP] {body} [SEP] 附件:{attachment}"
        else:
            text = f"{title} [SEP] {body}"

        label = int(row['tag'])

        # 分词和编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==================== 数据加载 ====================

def load_phishing_data(csv_path='data_set.csv'):
    """加载钓鱼邮件数据"""
    print(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)

    # 检查必需的列
    required_cols = ['title', 'body', 'tag']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 文件必须包含 '{col}' 列")

    # 清理数据
    df = df.dropna(subset=['title', 'body', 'tag'])
    df['tag'] = pd.to_numeric(df['tag'], errors='coerce')
    df = df.dropna(subset=['tag'])
    df['tag'] = df['tag'].astype(int)

    # 数据统计
    print(f"总样本数: {len(df)}")
    print(f"正常邮件 (tag=0): {len(df[df['tag'] == 0])}")
    print(f"钓鱼邮件 (tag=1): {len(df[df['tag'] == 1])}")

    # 检查数据有效性
    if len(df) == 0:
        raise ValueError("数据集为空！")

    normal_count = len(df[df['tag'] == 0])
    phishing_count = len(df[df['tag'] == 1])

    if normal_count == 0 or phishing_count == 0:
        raise ValueError(f"数据不平衡！正常邮件: {normal_count}, 钓鱼邮件: {phishing_count}")

    if normal_count < 2 or phishing_count < 2:
        raise ValueError("每个类别至少需要2个样本！")

    # 显示样本
    print("\n【数据示例】")
    print("\n正常邮件样本:")
    for idx, row in df[df['tag'] == 0].head(2).iterrows():
        print(f"  标题: {row['title'][:50]}...")
        print(f"  正文: {row['body'][:50]}...")
        if pd.notna(row.get('attachment_name')):
            print(f"  附件: {row['attachment_name']}")
        print()

    print("钓鱼邮件样本:")
    for idx, row in df[df['tag'] == 1].head(2).iterrows():
        print(f"  标题: {row['title'][:50]}...")
        print(f"  正文: {row['body'][:50]}...")
        if pd.notna(row.get('attachment_name')):
            print(f"  附件: {row['attachment_name']}")
        print()

    return df


# ==================== 训练函数 ====================

def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        # 释放内存
        del input_ids, attention_mask, labels, outputs
        if device.type == 'cpu':
            import gc
            gc.collect()

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估函数"""
    model.eval()
    predictions, true_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            del input_ids, attention_mask, labels, outputs
            if device.type == 'cpu':
                import gc
                gc.collect()

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1


# ==================== 主训练流程 ====================

def main():
    """主训练流程"""

    # 参数设置
    MODEL_NAME = './chinese-bert-wwm-ext'
    MAX_LENGTH = 512  # 邮件可能较长，增加到256
    BATCH_SIZE = 18  # 根据内存调整
    EPOCHS = 1
    LEARNING_RATE = 2e-5
    GRADIENT_ACCUMULATION_STEPS = 2

    print("=" * 60)
    print("钓鱼邮件检测模型训练")
    print("=" * 60)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型和分词器
    print("\nLoading model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # 二分类：0=正常，1=钓鱼
    )
    model.to(device)

    # 加载数据
    print("\nPreparing dataset...")
    df = load_phishing_data('data_set.csv')

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['tag']
    )

    print(f"\n训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")

    # 创建数据集
    train_dataset = PhishingEmailDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = PhishingEmailDataset(val_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练循环
    print("\nStarting training...")
    best_f1 = 0

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 50}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            GRADIENT_ACCUMULATION_STEPS
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, device
        )
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_phishing_detector.pt')
            print(f"✓ Saved best model with F1: {best_f1:.4f}")

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")

    # 保存完整模型
    model.save_pretrained('./phishing_email_detector')
    tokenizer.save_pretrained('./phishing_email_detector')
    print("Model saved to './phishing_email_detector'")

    # 最终评估
    print("\n" + "=" * 50)
    print("Final Evaluation on Validation Set")
    print("=" * 50)

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=['正常邮件', '钓鱼邮件'],
        digits=4
    ))


# ==================== 推理函数 ====================

def predict_email(title, body, attachment_name=None, model_path='./phishing_email_detector'):
    """
    预测邮件是否为钓鱼邮件

    Args:
        title: 邮件标题
        body: 邮件正文
        attachment_name: 附件名（可选）
        model_path: 模型路径

    Returns:
        dict: 预测结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 构建输入
    if attachment_name:
        text = f"{title} [SEP] {body} [SEP] 附件:{attachment_name}"
    else:
        text = f"{title} [SEP] {body}"

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    result = {
        'title': title,
        'is_phishing': bool(prediction),
        'prediction': '钓鱼邮件' if prediction == 1 else '正常邮件',
        'confidence': round(confidence, 4),
        'probabilities': {
            'normal': round(probs[0][0].item(), 4),
            'phishing': round(probs[0][1].item(), 4)
        }
    }

    return result


if __name__ == "__main__":
    # 训练模型
    main()

    # 测试推理
    print("\n" + "=" * 50)
    print("Testing predictions...")
    print("=" * 50)

    test_cases = [
        {
            'title': '会议通知',
            'body': '请各部门准时参加明天上午10点的工作会议。',
            'attachment': None
        },
        {
            'title': '【紧急】您的账户存在异常',
            'body': '您的银行账户检测到异常登录，请立即点击链接验证身份，否则将冻结账户。',
            'attachment': 'security_verify.exe'
        }
    ]

    for case in test_cases:
        result = predict_email(
            case['title'],
            case['body'],
            case['attachment']
        )
        print(f"\n标题: {result['title']}")
        print(f"预测: {result['prediction']}")
        print(f"置信度: {result['confidence'] * 100:.2f}%")