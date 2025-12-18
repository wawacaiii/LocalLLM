import requests
import json

class LocalLLMChat:
    def __init__(self, model_name="qwen3-vl:4b", api_url="http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.api_url = api_url
        self.history = []  # 保存对话上下文

    def chat(self, user_input):
        # 构建对话消息（包含上下文）
        self.history.append({"role": "user", "content": user_input})
        data = {
            "model": self.model_name,
            "messages": self.history,
            "stream": False,
            "temperature": 0.6
        }
        try:
            response = requests.post(self.api_url, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            assistant_response = result["message"]["content"]
            # 将模型响应加入上下文
            self.history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        except Exception as e:
            return f"对话失败：{str(e)}"

# 测试对话功能
if __name__ == "__main__":
    chatbot = LocalLLMChat()
    # 多轮对话测试（人工智能原理相关）
    while True:
        user_input = input("你：")
        if user_input.lower() in ["退出", "bye"]:
            print("模型：再见！")
            break
        response = chatbot.chat(user_input)
        print(f"模型：{response}")