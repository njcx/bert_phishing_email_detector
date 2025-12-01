#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
钓鱼邮件数据生成器 - 全面版
包含更多钓鱼邮件场景类型
"""

import csv
import random
import string


class ComprehensivePhishingEmailGenerator:
    def __init__(self):
        # 全面的邮件类型配置
        self.email_types = {
            # 银行金融类
            "bank_security": {
                "subject_template": "紧急：您的{bank_name}账户存在安全风险",
                "body_elements": ["异常登录", "安全验证", "账户冻结", "24小时内处理"],
                "attachment_templates": ["{bank_name}风险报告_{random}.pdf", 
                                       "{bank_name}安全证书_{random}.doc", 
                                       "{bank_name}账户验证_{random}.xls"],
                "platform": "银行",
                "keywords": ["银行", "账户", "冻结", "安全"]
            },
            
            "urgent_transfer": {
                "subject_template": "紧急：{company_name}财务部紧急转账请求",
                "body_elements": ["紧急转账", "资金调配", "时限操作", "保密通知"],
                "attachment_templates": ["转账申请表_{random}.pdf", 
                                       "资金调配单_{random}.doc", 
                                       "紧急通知_{random}.zip"],
                "platform": "财务",
                "keywords": ["转账", "财务", "紧急", "资金"]
            },
            
            "account_warning": {
                "subject_template": "重要：您的{platform_name}账户存在异常行为",
                "body_elements": ["账户异常", "违规操作", "立即申诉", "48小时处理"],
                "attachment_templates": ["账户报告_{random}.pdf", 
                                       "申诉材料_{random}.doc", 
                                       "违规记录_{random}.zip"],
                "platform": "账户",
                "keywords": ["异常", "违规", "账户", "警告"]
            },
            
            
            "fake_bill_refund": {
                "subject_template": "通知：{company_name}账单退款{amount}元待领取",
                "body_elements": ["账单退款", "系统错误", "立即领取", "7天内处理"],
                "attachment_templates": ["退款凭证_{random}.pdf", 
                                       "账单明细_{random}.xlsx", 
                                       "退款协议_{random}.doc"],
                "platform": "账单",
                "keywords": ["退款", "账单", "错误", "领取"]
            },
            
            # 电商购物类
            "order_issue": {
                "subject_template": "紧急：您的{company_name}订单存在{issue}问题",
                "body_elements": ["订单异常", "支付问题", "立即处理", "影响配送"],
                "attachment_templates": ["订单详情_{random}.pdf", 
                                       "异常报告_{random}.doc", 
                                       "处理单据_{random}.zip"],
                "platform": "电商",
                "keywords": ["订单", "问题", "异常", "处理"]
            },
            
            "logistics_abnormal": {
                "subject_template": "重要：您的快递包裹在{location}出现异常",
                "body_elements": ["包裹异常", "地址错误", "重新配送", "24小时确认"],
                "attachment_templates": ["物流报告_{random}.pdf", 
                                       "配送单据_{random}.doc", 
                                       "地址确认_{random}.zip"],
                "platform": "快递",
                "keywords": ["快递", "异常", "物流", "配送"]
            },
            
            
            # 社交平台类
            "account_security": {
                "subject_template": "警告：您的{platform_name}账号检测到{security_issue}",
                "body_elements": ["安全威胁", "账号保护", "立即验证", "密码重置"],
                "attachment_templates": ["安全报告_{random}.pdf", 
                                       "验证指南_{random}.doc", 
                                       "保护协议_{random}.zip"],
                "platform": "安全",
                "keywords": ["安全", "威胁", "保护", "验证"]
            },
            

            # 政府机构类
            "government_subsidy": {
                "subject_template": "通知：{gov_dept}{subsidy_name}补贴{amount}元",
                "body_elements": ["政府补贴", "政策扶持", "在线申请", "名额有限"],
                "attachment_templates": ["补贴申请表_{random}.pdf", 
                                       "政策文件_{random}.doc", 
                                       "申请指南_{random}.zip"],
                "platform": "政府",
                "keywords": ["政府", "补贴", "政策", "申请"]
            },
            
            "tax_refund": {
                "subject_template": "重要：{tax_dept}退税通知，退税金额{amount}元",
                "body_elements": ["个税退税", "系统推送", "银行到账", "15天内处理"],
                "attachment_templates": ["退税凭证_{random}.pdf", 
                                       "申报材料_{random}.doc", 
                                       "退税指南_{random}.zip"],
                "platform": "税务",
                "keywords": ["退税", "税务", "申报", "金额"]
            },
            
            # 企业内部类
            "internal_file": {
                "subject_template": "内部文件：{dept_name}重要文档待处理",
                "body_elements": ["内部文件", "部门文档", "保密级别", "限时处理"],
                "attachment_templates": ["{dept_name}文档_{random}.pdf", 
                                       "保密协议_{random}.doc", 
                                       "处理清单_{random}.zip"],
                "platform": "内部",
                "keywords": ["内部", "文档", "保密", "部门"]
            },
            
            "company_welfare": {
                "subject_template": "通知：{company_name}员工福利{coupon_type}申领",
                "body_elements": ["员工福利", "节日礼品", "在线申领", "截止日期"],
                "attachment_templates": ["福利申请表_{random}.pdf", 
                                       "领取指南_{random}.doc", 
                                       "礼品目录_{random}.zip"],
                "platform": "福利",
                "keywords": ["福利", "员工", "礼品", "申领"]
            },
            
            "holiday_subsidy": {
                "subject_template": "重要：{company_name}高温补贴{amount}元申领通知",
                "body_elements": ["高温补贴", "夏季关怀", "补贴申领", "政策通知"],
                "attachment_templates": ["补贴申请表_{random}.pdf", 
                                       "高温政策_{random}.doc", 
                                       "申领条件_{random}.zip"],
                "platform": "补贴",
                "keywords": ["高温", "补贴", "夏季", "关怀"]
            },
            
            # 系统通知类
            "email_storage": {
                "subject_template": "紧急：您的{email_provider}邮箱存储空间不足",
                "body_elements": ["存储不足", "系统升级", "立即处理", "影响收信"],
                "attachment_templates": ["存储报告_{random}.pdf", 
                                       "升级指南_{random}.doc", 
                                       "清理工具_{random}.zip"],
                "platform": "邮箱",
                "keywords": ["邮箱", "存储", "升级", "不足"]
            },
            
            "pending_task": {
                "subject_template": "提醒：{company_name}系统中您有待办事项",
                "body_elements": ["待办事项", "系统提醒", "限时处理", "逾期影响"],
                "attachment_templates": ["待办清单_{random}.pdf", 
                                       "处理流程_{random}.doc", 
                                       "提醒协议_{random}.zip"],
                "platform": "系统",
                "keywords": ["待办", "提醒", "系统", "处理"]
            },
            
            "compliance_notice": {
                "subject_template": "重要：{company_name}备案/合规检查通知",
                "body_elements": ["合规检查", "备案更新", "身份验证", "截止日期"],
                "attachment_templates": ["合规清单_{random}.pdf", 
                                       "备案材料_{random}.doc", 
                                       "检查指南_{random}.zip"],
                "platform": "合规",
                "keywords": ["合规", "备案", "检查", "身份"]
            },
            
            
            "lottery_activity": {
                "subject_template": "重磅活动：{company_name}年度抽奖{prize_name}大奖",
                "body_elements": ["活动抽奖", "幸运大奖", "限时参与", "奖品丰厚"],
                "attachment_templates": ["活动详情_{random}.pdf", 
                                       "抽奖规则_{random}.doc", 
                                       "奖品清单_{random}.zip"],
                "platform": "活动",
                "keywords": ["抽奖", "活动", "大奖", "奖品"]
            }
        }
        
        # 各种机构名称
        self.banks = ["招商银行", "工商银行", "建设银行", "农业银行", "中国银行", 
                     "平安银行", "交通银行", "民生银行", "浦发银行", "兴业银行"]
        
        self.companies = ["阿里巴巴", "腾讯", "百度", "字节跳动", "网易", "京东", 
                         "美团", "滴滴", "拼多多", "小米", "华为", "苹果", "三星"]
        
        self.platforms = ["微信", "QQ", "微博", "抖音", "快手", "知乎", 
                         "小红书", "钉钉", "支付宝", "淘宝", "天猫", "京东"]
        
        self.gov_depts = ["人社局", "民政局", "财政局", "税务局", "发改委", 
                         "工信部", "住建部", "教育部", "卫生部", "科技部"]
        
        self.subsidies = ["创业扶持", "就业补贴", "住房补贴", "教育补助", 
                         "医疗救助", "农业补贴", "中小企业扶持"]
        
        self.tax_depts = ["国家税务局", "地方税务局", "财政局", "税务机关"]
        
        self.departments = ["人力资源部", "财务部", "技术部", "市场部", "销售部", 
                           "运营部", "客服部", "法务部", "审计部", "采购部"]
        
        self.welfare_types = ["节日礼品", "生日券", "团建活动", "体检套餐", 
                             "购物券", "培训机会", "旅游券", "餐补"]
        
        self.email_providers = ["163邮箱", "126邮箱", "QQ邮箱", "新浪邮箱", 
                               "搜狐邮箱", "阿里邮箱", "网易邮箱"]
        
        self.systems = ["OA系统", "ERP系统", "CRM系统", "财务系统", "HR系统", 
                       "邮件系统", "VPN系统", "数据库", "网站后台", "APP后台"]
        
        self.operations = ["系统升级", "数据库优化", "安全补丁", "性能调优", 
                          "存储扩展", "网络维护", "备份恢复", "压力测试"]

    def generate_random_string(self, length=6):
        """生成随机字符串"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def get_institution_name(self, category):
        """根据类别获取机构名称"""
        if category == "bank":
            return random.choice(self.banks)
        elif category == "company":
            return random.choice(self.companies)
        elif category == "platform":
            return random.choice(self.platforms)
        elif category == "gov_dept":
            return random.choice(self.gov_depts)
        elif category == "tax_dept":
            return random.choice(self.tax_depts)
        elif category == "dept":
            return random.choice(self.departments)
        elif category == "email_provider":
            return random.choice(self.email_providers)
        elif category == "system":
            return random.choice(self.systems)
        return ""

    def generate_matching_email(self):
        """生成匹配的钓鱼邮件"""
        # 选择邮件类型
        email_type_key = random.choice(list(self.email_types.keys()))
        email_type = self.email_types[email_type_key]
        
        # 根据类型生成机构名称和相关变量
        bank_name = self.get_institution_name("bank")
        company_name = self.get_institution_name("company")
        platform_name = self.get_institution_name("platform")
        gov_dept = self.get_institution_name("gov_dept")
        tax_dept = self.get_institution_name("tax_dept")
        dept_name = self.get_institution_name("dept")
        email_provider = self.get_institution_name("email_provider")
        system_name = self.get_institution_name("system")
        
        # 生成相关变量
        random_amount = random.choice(["500", "1000", "2000", "5000", "10000", "15000", "25000"])
        random_issue = random.choice(["支付失败", "地址错误", "商品缺货", "物流延迟"])
        random_location = random.choice(["北京分拣中心", "上海转运站", "广州集散点", "深圳仓储"])
        random_promotion = random.choice(["满减券", "折扣券", "免费券", "积分券"])
        random_friend = random.choice(["张小明", "李小红", "王大伟", "陈小丽", "刘小华"])
        random_subsidy = random.choice(self.subsidies)
        random_welfare = random.choice(self.welfare_types)
        random_operation = random.choice(self.operations)
        random_security_issue = random.choice(["可疑登录", "密码泄露", "异常设备", "恶意攻击"])
        random_prize = random.choice(["iPhone 15", "MacBook Pro", "iPad", "AirPods"])
        
        # 生成匹配的主题
        subject_vars = {
            'bank_name': bank_name,
            'company_name': company_name,
            'platform_name': platform_name,
            'gov_dept': gov_dept,
            'subsidy_name': random_subsidy,
            'amount': random_amount,
            'tax_dept': tax_dept,
            'dept_name': dept_name,
            'coupon_type': random_welfare,
            'email_provider': email_provider,
            'system_name': system_name,
            'operation': random_operation,
            'location': random_location,
            'promotion_name': random_promotion,
            'friend_name': random_friend,
            'security_issue': random_security_issue,
            'prize_name': random_prize,
            'issue': random_issue
        }
        
        subject = email_type["subject_template"].format(**subject_vars)
        
        # 生成匹配的正文中内容
        body_element = random.choice(email_type["body_elements"])
        fake_link = f"http://verify-{self.generate_random_string()}.com/{email_type_key}"
        
        # 根据邮件类型生成对应正文
        if email_type_key == "bank_security":
            body = f"""尊敬的用户，

我们检测到您的{bank_name}账户在刚才存在异常登录行为。{body_element}，请立即进行安全验证。

验证链接：{fake_link}

如果您不在24小时内完成验证，您的账户将被冻结。

{bank_name}安全团队"""
            
        elif email_type_key == "urgent_transfer":
            body = f"""紧急通知：

{company_name}财务部需要紧急调配资金，金额：{random_amount}元。{body_element}，请立即处理此转账请求。

处理链接：{fake_link}

如不及时处理，将影响公司业务运行。

{company_name}财务部"""
            
        elif email_type_key == "account_warning":
            body = f"""账户警告：

您的{platform_name}账户检测到异常行为。{body_element}，请立即申诉并处理。

申诉链接：{fake_link}

如不处理，账户将在48小时内被限制。

{platform_name}客服"""
            
        elif email_type_key == "credit_loan_offer":
            body = f"""金融优惠通知：

{company_name}为您准备了专属的信用卡贷款优惠，额度：{random_amount}元。{body_element}，请立即申请。

申请链接：{fake_link}

优惠有效期至本月底。

{company_name}金融中心"""
            
        elif email_type_key == "fake_bill_refund":
            body = f"""账单退款通知：

由于系统错误，您的{company_name}账单产生退款，金额：{random_amount}元。{body_element}，请及时领取。

领取链接：{fake_link}

退款有效期7天，过期作废。

{company_name}财务部"""
            
        elif email_type_key == "order_issue":
            body = f"""订单异常通知：

您的{company_name}订单因{random_issue}存在处理问题。{body_element}，请及时处理。

处理链接：{fake_link}

如不及时处理，可能影响配送。

{company_name}客服"""
            
        elif email_type_key == "logistics_abnormal":
            body = f"""物流异常通知：

您的快递包裹在{random_location}出现异常。{body_element}，请确认地址信息。

确认链接：{fake_link}

如不及时处理，包裹将退回。

物流公司"""
            
        elif email_type_key == "promotion_win":
            body = f"""优惠券通知：

恭喜您获得{company_name}的{random_promotion}！{body_element}，请及时使用。

使用链接：{fake_link}

优惠券数量有限，先到先得。

{company_name}营销部"""
            
        elif email_type_key == "account_security":
            body = f"""安全威胁警告：

您的{platform_name}账号检测到{random_security_issue}。{body_element}，请立即验证身份。

验证链接：{fake_link}

如不验证，账号可能被恶意使用。

{platform_name}安全中心"""
            
        elif email_type_key == "friend_request":
            body = f"""好友请求通知：

{random_friend}通过{platform_name}向您发送了好友请求。{body_element}，请查看并处理。

查看链接：{fake_link}

请确认是否添加为好友。

{platform_name}"""
            
        elif email_type_key == "government_subsidy":
            body = f"""政府补贴通知：

{gov_dept}推出的{random_subsidy}政策现已开放申请，补贴金额：{random_amount}元。{body_element}，请尽快申请。

申请链接：{fake_link}

申请名额有限，先到先得。

{gov_dept}"""
            
        elif email_type_key == "tax_refund":
            body = f"""税务退税通知：

{tax_dept}的个税退税审批已完成，退税金额：{random_amount}元。{body_element}，请提供银行信息。

申请链接：{fake_link}

退税处理时间15天。

{tax_dept}"""
            
        elif email_type_key == "internal_file":
            body = f"""内部文件通知：

{dept_name}有重要文档需要您处理。{body_element}，请及时查看并处理。

查看链接：{fake_link}

文件级别：机密，请注意保密。

{dept_name}"""
            
        elif email_type_key == "company_welfare":
            body = f"""员工福利通知：

{company_name}为员工准备了{random_welfare}。{body_element}，请在线申领。

申领链接：{fake_link}

申领截止到月底。

{company_name}人力资源部"""
            
        elif email_type_key == "holiday_subsidy":
            body = f"""高温补贴通知：

{company_name}的夏季高温补贴现已开放申领，金额：{random_amount}元。{body_element}，请填写申请表。

申领链接：{fake_link}

申领条件请查看附件。

{company_name}"""
            
        elif email_type_key == "email_storage":
            body = f"""邮箱存储告警：

您的{email_provider}邮箱存储空间已不足。{body_element}，请立即处理。

处理链接：{fake_link}

不及时处理将影响正常收信。

{email_provider}系统"""
            
        elif email_type_key == "pending_task":
            body = f"""待办事项提醒：

{company_name}系统检测到您有待处理事项。{body_element}，请及时完成。

处理链接：{fake_link}

逾期可能影响工作进度。

{company_name}系统"""
            
        elif email_type_key == "compliance_notice":
            body = f"""合规检查通知：

{company_name}需要进行年度合规检查。{body_element}，请完成身份验证。

验证链接：{fake_link}

检查截止到月底。

{company_name}法务部"""
            
        elif email_type_key == "it_system_notice":
            body = f"""系统维护通知：

{system_name}将于今晚进行{random_operation}。{body_element}，请做好相关准备。

准备链接：{fake_link}

维护时间2小时。

IT部门"""
            
        elif email_type_key == "lottery_activity":
            body = f"""年度活动通知：

{company_name}推出{random_prize}大奖抽奖活动！{body_element}，请积极参与。

参与链接：{fake_link}

活动时间一周，机会难得。

{company_name}活动中心"""
        
        # 生成匹配的附件名
        attachment_template = random.choice(email_type["attachment_templates"])
        attachment_vars = {
            'random': self.generate_random_string(),
            'bank_name': bank_name,
            'company_name': company_name,
            'dept_name': dept_name
        }
        
        attachment = attachment_template.format(**attachment_vars)
        
        return {
            'subject': subject,
            'body': body,
            'attachment': attachment,
            'tag': 1
        }

    def generate_csv_data(self, num_emails=10000):
        """生成CSV格式的邮件数据"""
        data = []
        for i in range(num_emails):
            email_data = self.generate_matching_email()
            data.append([
                email_data['subject'],
                email_data['body'],
                email_data['attachment'],
                email_data['tag']
            ])
            if (i + 1) % 1000 == 0:
                print(f"已生成 {i + 1} 条邮件数据...")
        
        return data

def main():
    print("开始生成全面的钓鱼邮件训练数据...")
    print("目标：生成10000条包含20种场景的钓鱼邮件样本")
    print("新场景：IT系统通知、内部文件传递、紧急转账请求等")
    
    generator = ComprehensivePhishingEmailGenerator()
    
    # 生成数据
    emails_data = generator.generate_csv_data(10000)
    
    # 保存为CSV文件
    output_file = 'phishing_emails_dataset_semantic_10k.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['subject', 'body', 'attachment', 'tag'])
        writer.writerows(emails_data)
    
    print(f"全面钓鱼邮件数据生成完成！")
    print(f"文件保存路径：{output_file}")
    print(f"总共生成：{len(emails_data)} 条邮件数据")
    print("标签设置：1（表示钓鱼邮件）")
    print("包含20种钓鱼邮件场景类型")


if __name__ == "__main__":
    main()