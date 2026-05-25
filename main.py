from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from llm_service import chat_with_glm
from fastapi.middleware.cors import CORSMiddleware
from db_manager import LegalDB
import pandas as pd
import os
import uvicorn
import csv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document

app = FastAPI(title="法律咨询聊天机器人")

# 跨域
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# RAG 配置
MODEL_NAME = "qwen2.5:1.5b" 
VECTOR_DB_PATH = "./vector_db"

print(f"正在初始化本地模型: {MODEL_NAME}...")

try:
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    local_llm = OllamaLLM(
        model=MODEL_NAME, 
        temperature=0.1,
        num_ctx=2048
    )
except Exception as e:
    print(f"本地模型初始化失败: {e}")

# 向量库初始化
def init_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        print("加载现有向量库...")
        return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    
    print("正在读取 laws_structured.csv 并构建向量库...")
    texts = []
    try:
        if os.path.exists("laws_structured.csv"):
            with open("laws_structured.csv", "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" in row and row["text"].strip():
                        texts.append(row["text"].strip())
        
        if not texts:
            texts = ["试用期提前3天可离职", "公司应按月发放工资"]
            
        docs = [Document(page_content=t) for t in texts]
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print("向量库构建完成")
        return vs
    except Exception as e:
        print(f"向量库构建失败: {e}")
        return None

vector_store = init_vector_db()
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None

# 数据库
db = LegalDB()

class ChatRequest(BaseModel):
    question: str
    category: str = "general"
    user_id: str = "test_user_01"
    model_type: str = "local"  # 这里默认改成 local

# 新增这个接口，专门给本地模型用
@app.post("/api/chat")
async def chat_local(req: ChatRequest):
    response_data = {
        "code": 200,
        "answer": "抱歉，服务器暂时无法处理您的请求。"
    }
    try:
        history_data = db.get_history(req.user_id, req.category)
        formatted_history = [{"role": r['role'], "content": r['content']} for r in history_data]

        # 直接走本地模型逻辑
        if not retriever:
            response_data["answer"] = "本地检索服务未启动，请检查数据集。"
        else:
            docs = retriever.invoke(req.question)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"参考法条：\n{context}\n\n问题：{req.question}\n请简要回答："
            
            print(f"正在进行本地推理: {req.question}")
            try:
                raw_res = local_llm.invoke(prompt)
                response_data["answer"] = str(raw_res) if raw_res else "本地模型未生成内容。"
            except Exception as model_e:
                print(f"本地模型运行崩溃: {model_e}")
                response_data["answer"] = "⚠️ 本地显存不足，请清理后台程序或切换到 API 模式。"

        final_answer = response_data["answer"]
        db.save_chat_log(req.user_id, req.category, 'user', req.question)
        db.save_chat_log(req.user_id, req.category, 'assistant', final_answer)

        return response_data

    except Exception as e:
        print(f"本地模型接口错误: {e}")
        return {"code": 500, "answer": f"后端异常: {str(e)}"}


# 测试接口
@app.get("/")
def health_check():
    return {"status": "online", "model": MODEL_NAME}

@app.post("/api/chat/send")
async def chat_send(req: ChatRequest):
    response_data = {
        "code": 200,
        "answer": "抱歉，服务器暂时无法处理您的请求。",
        "mode": req.model_type
    }
    try:
        history_data = db.get_history(req.user_id, req.category)
        formatted_history = [{"role": r['role'], "content": r['content']} for r in history_data]

        # ✅ 修复：这里从 local_rag 改成 local
        if req.model_type == "local":
            if not retriever:
                response_data["answer"] = "本地检索服务未启动，请检查数据集。"
            else:
                docs = retriever.invoke(req.question)
                context = "\n".join([d.page_content for d in docs])
                prompt = f"参考法条：\n{context}\n\n问题：{req.question}\n请简要回答："
                
                print(f"正在进行本地推理: {req.question}")
                try:
                    raw_res = local_llm.invoke(prompt)
                    response_data["answer"] = str(raw_res) if raw_res else "本地模型未生成内容。"
                except Exception as model_e:
                    print(f"本地模型运行崩溃: {model_e}")
                    response_data["answer"] = "⚠️ 本地显存不足，请清理后台程序或切换到 API 模式。"
        else:
            # 在线GLM
            response_data["answer"] = chat_with_glm(req.question, formatted_history)

        final_answer = response_data["answer"]
        db.save_chat_log(req.user_id, req.category, 'user', req.question)
        db.save_chat_log(req.user_id, req.category, 'assistant', final_answer)

        return response_data

    except Exception as e:
        print(f"全局服务器错误: {e}")
        return {"code": 500, "answer": f"后端异常: {str(e)}", "mode": req.model_type}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)