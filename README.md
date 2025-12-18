# 说明
## 模型部署
安装ollama并下载模型，代码中使用的是qwen3-vl:4b，可自行更换。
## api调用
调用ollama的api/chat端口实现，模型可自行更换
## rag构建
使用lang_chain及faiss构建，支持docs/pdf/txt多种格式，知识库文件需放在./docs中，支持多个文件
