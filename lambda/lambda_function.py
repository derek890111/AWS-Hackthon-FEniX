import json
import hmac
import hashlib
import base64
import requests
import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import uuid
import random

# 從環境變數讀取配置
try:
    CHANNEL_ACCESS_TOKEN = os.environ['CHANNEL_ACCESS_TOKEN']
    CHANNEL_SECRET = os.environ['CHANNEL_SECRET']
    AGENT_ID = os.environ['AGENT_ID'] 
    AGENT_ALIAS_ID = os.environ['AGENT_ALIAS_ID']  # 從 Bedrock 控制台取得
    VOICE_TOKEN = os.environ['VOICE_TOKEN']
    BUCKET = os.environ['BUCKET']
    BUCKET_BASE = os.environ['BUCKET_BASE']
    VIDEO_BUCKET = os.environ['VIDEO_BUCKET']
except KeyError as e:
    print(f"Missing environment variable: {e}")
    raise

# 初始化 Bedrock Agent Runtime 客戶端
# 硬編碼區域為 us-west-2，因為 Agent 在此區域
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name='us-west-2'  # 直接指定區域
)

img2text_client = boto3.client('bedrock-runtime', region_name='us-west-2', config=Config(read_timeout=3 * 60))
s3_client = boto3.client('s3', region_name='us-west-2')
sessions = {}
def lambda_handler(event, context):
    try:
        # 取得 HTTP 請求的 body 和 headers
        if 'body' not in event or not event['body']:
            print("No body in event")
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'No body provided'})
            }
        
        body = event['body']
        signature = event.get('headers', {}).get('x-line-signature', '')

        # # 驗證 LINE Webhook 簽章
        if not verify_signature(body, signature):
            print("Invalid signature")
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Invalid signature'})
            }

        # 解析 LINE 事件
        try:
            body_json = json.loads(body)
            events = body_json.get('events', [])
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Invalid JSON format'})
            }

        # 處理每個事件
        for event in events:
            if event['type'] == 'message' and event['message']['type'] == 'text':
                reply_token = event['replyToken']
                user_message = event['message']['text']
                user_id = event['source']['userId']
                #Handle "猜猜我是誰"
                print(user_message)
                # 調用 Bedrock Agent 生成回應
                agent_response = invoke_bedrock_agent(user_message)
                messages = [{
                    "type": "text",
                    "text": agent_response
                }]
                send_reply(reply_token, messages)
                
                if user_message == "猜猜我是誰":
                    
                    sessions[user_id] = {"audio1": None, "audio2": None, "correct_answer": None}
                    audio_uris, chosen_speaker = guess_voice_uri(user_message)
                    if audio_uris == "error":
                        messages = [{"type": "text", "text": f"Error generating audio: {chosen_speaker}"}]
                        send_push(user_id, messages)
                        continue
                    sessions[user_id]["audio1"] = audio_uris[0]
                    sessions[user_id]["audio2"] = audio_uris[1]
                    sessions[user_id]["correct_answer"] = random.randint(1,2)
                    messages = [
                        {"type": "audio", "originalContentUrl": audio_uris[0], "duration": 3000},
                        {"type": "audio", "originalContentUrl": audio_uris[1], "duration": 3000}
                    ]
                    send_push(user_id, messages)
                    continue

                elif "生日快樂" in user_message:
                    object_key = "jiachi_short_HBD.mp4"
                    url = create_tmp_url(VIDEO_BUCKET, object_key)
                    previewUrl = create_tmp_url(BUCKET_BASE,"488722645_18493729093045455_5564924153951444253_n.jpg")
                    messages=[{
                        "type": "video",
                        "originalContentUrl": url,
                        "previewImageUrl": previewUrl
                    }]
                    send_push(user_id, messages)
                    continue
                    
                elif "新年快樂" in user_message:
                    object_key = "jiachi_short_HNY.mp4"
                    url = create_tmp_url(VIDEO_BUCKET, object_key)
                    previewUrl = create_tmp_url(BUCKET_BASE,"488722645_18493729093045455_5564924153951444253_n.jpg")
                    messages=[{
                        "type": "video",
                        "originalContentUrl": url,
                        "previewImageUrl": previewUrl
                    }]
                    send_push(user_id, messages)
                    continue
                    
                gamania_response = get_voice_uri(agent_response)
                messages = []
                if gamania_response["status"] == "success":
                    mp3_uri = gamania_response["uri"]
                    status, duration = get_mp3_duration_from_header(mp3_uri)

                    if status == 200:
                        messages.append({
                            "type": "audio",
                            "originalContentUrl": gamania_response["uri"],
                            "duration": duration
                        })
                    else:
                        messages.append({
                            "type": "text",
                            "text": gamania_response["uri"]
                        })
                else:
                    messages.append({
                        "type": "text",
                        "text": gamania_response["message"]
                    })
                send_push(user_id, messages)

                status, url = gen_image(user_message, agent_response)
                if not status:
                    raise Exception(url)
                if status:
                    send_push(user_id, [{
                    "type": "image",
                    "originalContentUrl": url,  
                    "previewImageUrl": url
                    }])
                else:
                    send_push(user_id, [{
                    "type": "text",
                    "text": url
                    }])
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Success'})
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Server error: {str(e)}'})
        }

# generate voice url 
def guess_voice_uri(text):
    speakers = ["chiachi", "max"]
    chosen_speaker = random.choice(speakers)
    model_id = 4 if chosen_speaker == "chiachi" else 2
    response1 = get_voice_uri(text, chosen_speaker, model_id)
    other_speaker = speakers[0] if chosen_speaker == speakers[1] else speakers[1]
    model_id_other = 4 if other_speaker == "chiachi" else 2
    response2 = get_voice_uri(text, other_speaker, model_id_other)
    if response1["status"] == "success" and response2["status"] == "success":
        return [response1["uri"], response2["uri"]], chosen_speaker
    else:
        return "error", None

# create tmp url
def create_tmp_url(bucket, object_key):
    try:
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket,
                'Key': object_key
            },
            ExpiresIn=3600  # 3600秒 = 1小時
        )
        return presigned_url
    except Exception as e:
        return str(e)

# gen image
def gen_image(user_message, response_message):
    try:
        
        response = s3_client.get_object(Bucket=BUCKET_BASE, Key="488722645_18493729093045455_5564924153951444253_n.jpg")
        image_bytes = response['Body'].read()
        reference_image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        system_prompt = """You are assisting in creating personalized visual descriptions for a fan interaction chatbot.
Each input includes a fan and idol conversation in Chinese.
Based on the emotional tone, suggest an appropriate setting and clothing style to match the atmosphere.
Do not change physical appearance details. Focus only on background scenery or outfit themes.
Keep your description natural, positive, and concise."""
        image_des_prompt = "A male idol wearing a black sleeveless top and a silver chain necklace, standing in front of a white background, with short curly hair and natural makeup."
        movement_prompt_message = f"{system_prompt} user said: {user_message}, idol said: {response_message}. image describtion:{image_des_prompt}. The total length of the output must not exceed 480 characters."
        print(movement_prompt_message)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "text": movement_prompt_message
                    }]
                }
            ],
        }
        response = img2text_client.invoke_model(
            body=json.dumps(payload),
            modelId="arn:aws:bedrock:us-west-2:305345298333:inference-profile/us.amazon.nova-pro-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        movement_prompt = json.loads(response.get("body").read())
        print(movement_prompt)
        movement_prompt = movement_prompt['output']['message']['content'][0]['text']
        # prompt = "Keeping the original face, hairstyle, and pose of the young male idol. Change the background to a sunlit rooftop with clear blue skies, and update the outfit to a casual white hoodie and jeans. The atmosphere remains bright, fresh, and uplifting."
        body = json.dumps(
                {
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {
                    "text": movement_prompt,
                    "images": [
                        reference_image_base64
                    ],  # May provide up to 5 reference images here
                    "similarityStrength": 0.7,  # How strongly the input images influence the output. From 0.2 through 1.
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,  # Number of images to generate, up to 5.
                    "cfgScale": 6.5,  # How closely the prompt will be followed
                    "seed": 42,  # Any number from 0 through 858,993,459
                    "quality": "standard",  # Either "standard" or "premium". Defaults to "standard".
                },
            }
        )
        response = img2text_client.invoke_model(
            body=body,
            modelId="amazon.titan-image-generator-v1",
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())
        base64_images = response_body.get("images")
        if isinstance(base64_images, list):
            base64_images = base64_images[0]

        image_bytes = base64.b64decode(base64_images)
        object_key = f"generated-images/{uuid.uuid4()}.png"

        s3_client.put_object(
            Bucket=BUCKET,
            Key=object_key,
            Body=image_bytes,
            ContentType='image/png'
        )
        file_url = f"https://{BUCKET}.s3.amazonaws.com/{object_key}"
        tmp_url = create_tmp_url(BUCKET, object_key)
        return True, tmp_url
    except Exception as e:
        return False, str(e)


# mp3 duration
def get_mp3_duration_from_header(url):
    try:
        headers = {"Range": "bytes=0-2000"}  # 只抓前2KB就好，夠讀header
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        content = response.content

        # 找 MPEG Audio Frame Header (簡化版做法)
        # MPEG header一般是 0xFFFB 或 0xFFF3 或 0xFFF2 開頭
        for i in range(len(content) - 1):
            if content[i] == 0xFF and (content[i+1] & 0xE0) == 0xE0:
                # 抓 bitrate index
                bitrate_index = (content[i+2] >> 4) & 0x0F
                # 抓 sampling rate index
                sampling_rate_index = (content[i+2] >> 2) & 0x03

                # 參考表（只做常見MPEG1 Layer III）
                bitrate_table = [
                    None, 32, 40, 48, 56, 64, 80, 96,
                    112, 128, 160, 192, 224, 256, 320, None
                ]  # in kbps
                sampling_rate_table = [44100, 48000, 32000, None]

                bitrate = bitrate_table[bitrate_index] * 1000  # bps
                sampling_rate = sampling_rate_table[sampling_rate_index]

                if bitrate and sampling_rate:
                    # 再去抓整個檔案大小
                    head = requests.head(url)
                    file_size_bytes = int(head.headers['Content-Length'])
                    file_size_bits = file_size_bytes * 8

                    duration_sec = file_size_bits / bitrate
                    duration_ms = int(duration_sec * 1000)
                    return 200, duration_ms
                break
        # 如果沒抓到header，回預設
        return 404, 3000
    except Exception as e:
        print(f"Failed to parse mp3 header: {e}")
        return 404, 3000

# 驗證 LINE Webhook 簽章
def verify_signature(body, signature):
    try:
        body_bytes = body.encode('utf-8')
        hash = hmac.new(CHANNEL_SECRET.encode('utf-8'), body_bytes, hashlib.sha256)
        computed_signature = base64.b64encode(hash.digest()).decode('utf-8')
        return computed_signature == signature
    except Exception as e:
        print(f"Signature verification error: {e}")
        return False

# 調用 Bedrock Agent
def invoke_bedrock_agent(prompt):
    try:
        # 生成唯一 session ID（可重複使用以維持對話上下文）
        session_id = str(uuid.uuid4())
        
        # 調用 Bedrock Agent
        response = bedrock_agent_runtime.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=prompt
        )

        # 解析回應
        completion = ""
        for chunk in response['completion']:
            if 'chunk' in chunk:
                completion += chunk['chunk']['bytes'].decode('utf-8')
            elif 'trace' in chunk:
                print(f"Trace: {chunk['trace']}")
        
        return completion if completion else "No response from Agent"
    except ClientError as e:
        print(f"Bedrock Agent error: {e}")
        return f"Error invoking Agent: {str(e)}"
    except Exception as e:
        print(f"Error invoking Agent: {e}")
        return f"Error: {str(e)}"

def get_voice_uri(text, speaker_name="chiachi", model_id=4):
    url = "https://persona-sound.data.gamania.com/api/v1/public/voice"
    headers = {
        "Authorization": f"Bearer {VOICE_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    params = {
        "text": text,
        "model_id": model_id,
        "speaker_name": speaker_name,
        "speed_factor": 1,
        "mode": "file"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return {"status": "success", "uri": data["media_url"]}
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "message": f"HTTP error: {e}, response: {response.text}"}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"JSON decode error: {e}, response: {response.text}"}
    except KeyError as e:
        return {"status": "error", "message": f"KeyError: {e}, response: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {e}, response: {response.text}"}
    

# 回傳訊息到 LINE
def send_reply(reply_token, messages):
    url = 'https://api.line.me/v2/bot/message/reply'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {CHANNEL_ACCESS_TOKEN}'
    }
    for i in range(len(messages)):
        if messages[i]["type"] == "text":
            messages[i]["text"] = messages[i]["text"][:2000]
    data = {
        'replyToken': reply_token,
        'messages': messages
        # 'messages': [{
        #     'type': 'text',
        #     'text': message[:2000]  # LINE 訊息長度限制
        # }]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send reply: {e}")

# push messages to certain userid
def send_push(user_id, messages):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {CHANNEL_ACCESS_TOKEN}'
    }
    data = {
        'to': user_id,
        'messages': messages
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send push: {e}")
