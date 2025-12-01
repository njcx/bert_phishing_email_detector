import requests

# API 基础URL
BASE_URL = "http://127.0.0.1:5002"

# API Key 认证
API_KEY = "H8jR4nPqW6sYtVcE"  # 与服务器配置的密钥一致

# 请求头
HEADERS = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
}

normal_emails = [
    # 工作邮件
    {
        'title': '周例会通知',
        'body': '各位同事，本周例会将于周五下午3点在会议室召开，请准时参加。',
        'attachment': None
    },
    {
        'title': '项目进度报告',
        'body': '附件是本月项目进度报告，请查收并提出意见。',
        'attachment': 'project_report_202401.pdf'
    },
    {
        'title': 'Re: 关于系统升级的建议',
        'body': '感谢您的建议，我们会在下次技术评审会议上讨论这个方案。',
        'attachment': None
    },
    {
        'title': '培训课程安排',
        'body': '下周三将举办Python编程培训，有兴趣的同事请在系统中报名。',
        'attachment': 'training_schedule.xlsx'
    },

    # 商务邮件
    {
        'title': '合作意向书',
        'body': '尊敬的合作伙伴，感谢您对我们公司的信任，附件是合作意向书，请查阅。',
        'attachment': 'cooperation_agreement.docx'
    },
    {
        'title': '发票已开具',
        'body': '您好，您订购的产品发票已开具，请在附件中查看。',
        'attachment': 'invoice_20240115.pdf'
    },
    {
        'title': '订单确认',
        'body': '您的订单已确认，预计3-5个工作日内发货，物流单号稍后发送。',
        'attachment': 'order_confirmation.pdf'
    },

    # 通知邮件
    {
        'title': '系统维护通知',
        'body': '为提升服务质量，系统将于本周日凌晨2-6点进行维护，期间暂停服务。',
        'attachment': None
    },
    {
        'title': '活动报名确认',
        'body': '您已成功报名参加年会活动，活动详情请查看附件。',
        'attachment': 'event_details.pdf'
    },
    {
        'title': '账单提醒',
        'body': '您本月的账单已生成，请及时查看并按时缴费。',
        'attachment': 'monthly_bill.pdf'
    },

    # 个人邮件
    {
        'title': '旅游照片分享',
        'body': '上周旅游的照片整理好了，分享给你看看。',
        'attachment': 'travel_photos.zip'
    },
    {
        'title': '生日聚会邀请',
        'body': '下周六是我的生日，诚邀你参加生日聚会，地点在xxx餐厅。',
        'attachment': None
    },
]

# ==================== 钓鱼邮件模板 ====================

phishing_emails = [
    # 银行类钓鱼
    {
        'title': '【紧急】工商银行账户异常通知',
        'body': '尊敬的客户，您的账户检测到异常交易，请立即点击链接验证身份：http://icbc-verify.com/login 否则将冻结账户。',
        'attachment': 'security_verify.exe'
    },
    {
        'title': '中国银行：您的账户已被锁定',
        'body': '系统检测到您的账户存在风险，已被临时锁定。请立即访问 http://boc-unlock.net 解锁。',
        'attachment': None
    },

    # 电商类钓鱼
    {
        'title': '【淘宝】订单异常，请立即处理',
        'body': '您的订单出现异常，商品无法发货。请联系客服QQ：123456789 或点击链接处理：http://taobao-help.cc',
        'attachment': 'order_problem.html'
    },

    {
        'title': '拼多多中奖通知',
        'body': '恭喜您中奖了！奖金10000元，请点击领取：http://pdd-prize.top 需要提供银行卡信息。',
        'attachment': None
    },

    # 支付类钓鱼
    {
        'title': '支付宝安全提醒',
        'body': '您的支付宝账户存在安全风险，请立即验证身份信息，否则将限制使用。验证地址：http://alipay-safe.cn',
        'attachment': 'identity_verify.exe'
    },
    {
        'title': '微信支付异常',
        'body': '您的微信支付功能已被限制，原因：异常交易。请下载附件解除限制。',
        'attachment': 'wechat_unlock.apk'
    },

    # 政府/公共服务类钓鱼
    {
        'title': '【税务局】个人所得税退税通知',
        'body': '根据最新政策，您可申请退税2580元。请访问：http://tax-refund.gov.cc 填写银行信息。',
        'attachment': 'tax_refund_guide.pdf'
    },
    {
        'title': '社保中心：补缴通知',
        'body': '您有社保欠费需补缴，请点击链接查看详情并缴费：http://social-security.org.cn',
        'attachment': 'payment_notice.doc'
    },

    {
        'title': '【IT部门】系统密码重置',
        'body': '系统升级要求所有员工重置密码，请访问：http://company-reset.com 完成重置。',
        'attachment': None
    },

    # 快递/物流类钓鱼
    {
        'title': '顺丰速运：您的包裹无法投递',
        'body': '您的快递因地址不详无法投递，请点击链接更新地址：http://sf-express.cc 并支付5元重新配送费。',
        'attachment': None
    },

]


response = requests.post(
    BASE_URL+'/detect/batch',
    json={
        'emails': normal_emails
    },
    headers=HEADERS
)
print(normal_emails)
summary = response.json()['data']['summary']
print(f"钓鱼邮件: {summary['phishing_count']} / {summary['total']}")
print(f"钓鱼率: {summary['phishing_rate']}%")


response = requests.post(
    BASE_URL+'/detect/batch',
    json={
        'emails': phishing_emails
    },
    headers=HEADERS
)
print(phishing_emails)
summary = response.json()['data']['summary']
print(f"钓鱼邮件: {summary['phishing_count']} / {summary['total']}")
print(f"钓鱼率: {summary['phishing_rate']}%")


for email in normal_emails:
    response = requests.post(
        BASE_URL+'/detect',
        json=email,
        headers=HEADERS
    )
    print(email)

    result = response.json()
    print(f"是否钓鱼: {result['data']['is_phishing']}")
    print(f"风险等级: {result['data']['risk_level']}")


for email in phishing_emails:
    print(email)
    response = requests.post(
        BASE_URL + '/detect',
        json=email,
        headers=HEADERS
    )

    result = response.json()
    print(f"是否钓鱼: {result['data']['is_phishing']}")
    print(f"风险等级: {result['data']['risk_level']}")