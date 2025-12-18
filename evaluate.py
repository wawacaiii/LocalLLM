import json
import time
import jieba  # 需安装：pip install jieba
from sklearn.metrics.pairwise import cosine_similarity
from rag_build_kb import RAGConfig, rag_qa
from langchain_ollama import OllamaEmbeddings

# ====================== 1. 评估指标定义 ======================

class RAGEvaluator:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.metrics = {
            "recall": 0.0,  # 检索召回率（是否找到相关文档）
            "precision": 0.0,  # 回答准确率（是否包含核心信息）
            "avg_response_time": 0.0
        }
        # 初始化嵌入模型，用于语义相似度计算
        self.embeddings = OllamaEmbeddings(model=self.config.EMBED_MODEL)

    def _keyword_match(self, answer: str, ground_truth: str) -> float:
        """
        关键词匹配：计算答案与标准答案的核心词重合率
        返回值：0~1（1=完全命中，0=无命中）
        """
        # 分词（中文适配）
        answer_words = set(jieba.lcut(answer))
        truth_words = set(jieba.lcut(ground_truth))

        # 过滤停用词（无意义的词）
        stop_words = {"的", "是", "在", "有", "我", "也", "不", "了", "和", "为"}
        answer_words = {w for w in answer_words if w not in stop_words and len(w) > 1}
        truth_words = {w for w in truth_words if w not in stop_words and len(w) > 1}

        if not truth_words:
            return 1.0 if answer.strip() else 0.0

        # 计算命中的核心词比例
        match_count = len(answer_words & truth_words)
        return match_count / len(truth_words)

    def _semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """
        语义相似度：计算答案与标准答案的向量余弦相似度
        返回值：0~1（1=语义完全一致，0=无关联）
        """
        # 生成向量
        answer_emb = self.embeddings.embed_query(answer.strip() or "无回答")
        truth_emb = self.embeddings.embed_query(ground_truth)

        # 计算余弦相似度
        similarity = cosine_similarity([answer_emb], [truth_emb])[0][0]
        return max(0.0, min(1.0, similarity))  # 限制在0~1之间

    def evaluate_single_question(self, question: str, ground_truth: str) -> dict:
        """单问题评估（柔性指标）"""
        start_time = time.time()
        answer = rag_qa(question, self.config)
        response_time = time.time() - start_time

        # 处理空回答
        if not answer.strip():
            answer = "未在知识库中找到相关答案（模型无输出）"

        # 1. 召回率：语义相似度≥0.5 视为召回成功
        semantic_sim = self._semantic_similarity(answer, ground_truth)
        recall = 1 if semantic_sim >= 0.5 else 0

        # 2. 准确率：关键词匹配率≥0.5 视为准确
        keyword_match = self._keyword_match(answer, ground_truth)
        precision = 1 if keyword_match >= 0.5 else 0

        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "semantic_similarity": round(semantic_sim, 2),  # 语义相似度（新增）
            "keyword_match_rate": round(keyword_match, 2),  # 关键词匹配率（新增）
            "recall": recall,
            "precision": precision,
            "response_time": round(response_time, 2)
        }

    def batch_evaluate(self, eval_dataset_path: str) -> dict:
        """批量评估（优化版）"""
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            eval_dataset = json.load(f)

        total_recall = 0
        total_precision = 0
        total_time = 0
        total_sem_sim = 0
        total_keyword_match = 0
        results = []

        for idx, item in enumerate(eval_dataset):
            print(f"评估第 {idx + 1}/{len(eval_dataset)} 个问题：{item['question']}")
            res = self.evaluate_single_question(item["question"], item["ground_truth"])
            results.append(res)
            total_recall += res["recall"]
            total_precision += res["precision"]
            total_time += res["response_time"]
            total_sem_sim += res["semantic_similarity"]
            total_keyword_match += res["keyword_match_rate"]

        # 计算平均指标（新增语义/关键词指标）
        n = len(eval_dataset)
        self.metrics["recall"] = round(total_recall / n, 2)
        self.metrics["precision"] = round(total_precision / n, 2)
        self.metrics["avg_response_time"] = round(total_time / n, 2)
        self.metrics["avg_semantic_similarity"] = round(total_sem_sim / n, 2)
        self.metrics["avg_keyword_match_rate"] = round(total_keyword_match / n, 2)

        # 保存评估结果（包含新增指标）
        with open("rag_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "summary_metrics": self.metrics,
                "detailed_results": results
            }, f, ensure_ascii=False, indent=2)

        # 打印优化后的评估报告
        print("\n===== 优化版评估结果 =====")
        print(f"平均召回率：{self.metrics['recall']}")
        print(f"平均准确率：{self.metrics['precision']}")
        print(f"平均语义相似度：{self.metrics['avg_semantic_similarity']}")
        print(f"平均关键词匹配率：{self.metrics['avg_keyword_match_rate']}")
        print(f"平均响应时间：{self.metrics['avg_response_time']}s")

        return self.metrics


# ====================== 2. 优化策略函数 ======================
def optimize_retrieval_strategy(config: RAGConfig, strategy: str = "similarity_score_threshold"):
    """优化召回策略"""
    from rag_build_kb import OllamaEmbeddings, FAISS

    # 加载向量库
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    db = FAISS.load_local(
        config.VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 策略1：基于分数阈值的检索（只召回相似度≥阈值的文档）
    if strategy == "similarity_score_threshold":
        retriever = db.as_retriever(
            search_kwargs={"k": config.TOP_K, "score_threshold": 0.7}  # 阈值可调整
        )
    # 策略2：MMR（最大边际相关性，减少冗余）
    elif strategy == "mmr":
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.TOP_K, "fetch_k": 20}  # 先取20个再选5个
        )
    # 策略3：调整Top-K数量
    elif strategy == "adjust_topk":
        retriever = db.as_retriever(search_kwargs={"k": 8})  # 增大Top-K

    return retriever


def optimize_prompt_template():
    """优化提示词（进阶版）"""
    from langchain_core.prompts import ChatPromptTemplate

    advanced_prompt = """
    任务：基于参考文档回答用户问题，要求如下：
    1. 相关性：仅使用参考文档中的信息，禁止使用外部知识；
    2. 完整性：覆盖问题的所有子问题，不要遗漏关键信息；
    3. 准确性：严格匹配文档内容，避免概括或推测；
    4. 格式：
       - 答案主体：简洁明了，分点说明（如有必要）；
       - 来源标注：在答案末尾注明“来源：[文件名]”；
       - 无答案时：仅返回“未在知识库中找到相关答案”。

    参考文档：
    {context}

    用户问题：{question}

    最终回答：
    """
    return ChatPromptTemplate.from_template(advanced_prompt)


# ====================== 3. 评估示例 ======================
if __name__ == "__main__":
    # 1. 准备非公开评估数据集（eval_dataset.json）
    # 格式示例：
    # [
    #   {"question": "数据集核心结论是什么？", "ground_truth": "核心结论是XXX"},
    #   {"question": "文档中提到的方法有哪些？", "ground_truth": "方法包括A、B、C"}
    # ]

    # 2. 执行批量评估
    evaluator = RAGEvaluator()
    evaluator.batch_evaluate("eval_dataset.json")

    # 3. 测试优化策略
    # optimized_retriever = optimize_retrieval_strategy(RAGConfig(), "mmr")
    # optimized_prompt = optimize_prompt_template()