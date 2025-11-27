from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
from functools import wraps
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 全局变量
model = None
tokenizer = None
device = None

# 配置
MODEL_PATH = './phishing_email_detector'
MAX_LENGTH = 256
CONFIDENCE_THRESHOLD = 0.5
API_KEY = ["H8jR4nPqW6sYtVcE", "T7gFpN3mKx9LcVbQ", "R2vZqW8nJk4PmSxH"]


def load_model():
    """加载模型"""
    global model, tokenizer, device

    try:
        logger.info("Loading phishing detection model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def require_api_key(f):
    """API Key 认证"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not api_key:
            return jsonify({
                'success': False,
                'error': 'Missing API key'
            }), 401

        if api_key not in API_KEY:
            return jsonify({
                'success': False,
                'error': 'Invalid API key'
            }), 403

        return f(*args, **kwargs)

    return decorated_function


def predict_phishing(title, body, attachment_name=None, threshold=CONFIDENCE_THRESHOLD):
    """预测邮件是否为钓鱼邮件"""
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    # 构建输入文本
    if attachment_name and str(attachment_name).strip():
        text = f"{title} [SEP] {body} [SEP] 附件:{attachment_name}"
    else:
        text = f"{title} [SEP] {body}"

    # 编码
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    is_phishing = prediction == 1 and confidence >= threshold

    return {
        'title': title,
        'is_phishing': is_phishing,
        'prediction': 'phishing' if prediction == 1 else 'normal',
        'confidence': round(confidence, 4),
        'probabilities': {
            'normal': round(probs[0][0].item(), 4),
            'phishing': round(probs[0][1].item(), 4)
        },
        'risk_level': get_risk_level(probs[0][1].item())
    }


def get_risk_level(phishing_prob):
    """根据钓鱼概率返回风险等级"""
    if phishing_prob < 0.3:
        return 'low'
    elif phishing_prob < 0.7:
        return 'medium'
    else:
        return 'high'


@app.route('/', methods=['GET'])
def index():
    """首页"""
    return jsonify({
        'status': 'ok',
        'message': 'Phishing Email Detection API',
        'version': '1.0.0',
        'model_loaded': model is not None
    })


@app.route('/detect', methods=['POST'])
@require_api_key
def detect():
    """
    检测单个邮件

    请求体：
    {
        "title": "邮件标题",
        "body": "邮件正文",
        "attachment_name": "附件名.exe",  # 可选
        "threshold": 0.5  # 可选
    }

    响应：
    {
        "success": true,
        "data": {
            "title": "邮件标题",
            "is_phishing": true,
            "prediction": "phishing",
            "confidence": 0.9567,
            "probabilities": {
                "normal": 0.0433,
                "phishing": 0.9567
            },
            "risk_level": "high"
        }
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503

        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # 验证必需字段
        if 'title' not in data or 'body' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: title and body'
            }), 400

        title = data['title']
        body = data['body']
        attachment = data.get('attachment_name')
        threshold = data.get('threshold', CONFIDENCE_THRESHOLD)

        # 验证输入
        if not isinstance(title, str) or not isinstance(body, str):
            return jsonify({
                'success': False,
                'error': 'title and body must be strings'
            }), 400

        if len(title.strip()) == 0 or len(body.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'title and body cannot be empty'
            }), 400

        # 执行预测
        start_time = time.time()
        result = predict_phishing(title, body, attachment, threshold)
        elapsed_time = (time.time() - start_time) * 1000

        return jsonify({
            'success': True,
            'data': result,
            'processing_time_ms': round(elapsed_time, 2)
        }), 200

    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detect/batch', methods=['POST'])
@require_api_key
def detect_batch():
    """
    批量检测邮件

    请求体：
    {
        "emails": [
            {
                "title": "标题1",
                "body": "正文1",
                "attachment_name": "附件1"
            },
            {
                "title": "标题2",
                "body": "正文2"
            }
        ],
        "threshold": 0.5
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503

        data = request.get_json()

        if not data or 'emails' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: emails'
            }), 400

        emails = data['emails']
        threshold = data.get('threshold', CONFIDENCE_THRESHOLD)

        if not isinstance(emails, list) or len(emails) == 0:
            return jsonify({
                'success': False,
                'error': 'emails must be a non-empty list'
            }), 400

        if len(emails) > 50:
            return jsonify({
                'success': False,
                'error': 'Too many emails (max 50 per request)'
            }), 400

        # 批量预测
        start_time = time.time()
        results = []
        phishing_count = 0

        for email in emails:
            if not isinstance(email, dict):
                results.append({'error': 'Invalid email format'})
                continue

            title = email.get('title', '')
            body = email.get('body', '')
            attachment = email.get('attachment_name')

            if not title or not body:
                results.append({'error': 'Missing title or body'})
                continue

            result = predict_phishing(title, body, attachment, threshold)
            results.append(result)

            if result['is_phishing']:
                phishing_count += 1

        elapsed_time = (time.time() - start_time) * 1000

        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'summary': {
                    'total': len(results),
                    'phishing_count': phishing_count,
                    'normal_count': len(results) - phishing_count,
                    'phishing_rate': round(phishing_count / len(results) * 100, 2) if results else 0
                }
            },
            'processing_time_ms': round(elapsed_time, 2)
        }), 200

    except Exception as e:
        logger.error(f"Batch detection error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # 启动时加载模型
    if not load_model():
        logger.error("Failed to load model. Exiting...")
        exit(1)

    # 启动服务
    logger.info("Starting Phishing Detection API...")
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False,
        threaded=True
    )