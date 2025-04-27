# AWS-HACKTHON, 眾量橘

## Description


## System Architecture
![image]()

## External Packages
Public Voice API
- Documentation: [PublicVoiceAPI-FEniX.pdf](https://reurl.cc/4L1MEK)
- Response Examples:
  - mode=stream:
    ```json
    {
      "message": "success",
      "media_url": "https://cdn.data.gamania.com/persona-sound/20241014/ting/05870db4-6b07-48a0-b7f0-3ed69e137989.wav"
    }
    ```
  - mode=file: Binary file content


## DataFlow：AI 偶像訊息處理流程


1. 使用者 → LINE 傳送訊息  
2. LINE → API Gateway  
3. API Gateway → Lambda
4. Lambda → Nova-Content 生成文字回覆  
5. Lambda → 呼叫 Public Voice API 進行語音合成  
6. Lambda → 回傳「文字 + 音訊 URL」給 API Gateway → LINE → 用戶收到語音訊息



