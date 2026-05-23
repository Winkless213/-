RAG_SYSTEM_PROMPT = """你是一个企业知识库助手。根据提供的参考资料回答用户问题。

规则：
1. 只根据参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，明确说"根据现有资料，我无法回答这个问题"
3. 回答要简洁准确
4. 引用来源时说明出自哪个文档"""

RAG_USER_TEMPLATE = """参考资料：
{context}

用户问题：{question}

请根据参考资料回答。"""


def build_rag_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Build messages for RAG query."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("filename", "未知来源")
        context_parts.append(f"[{i}] 来源: {source}\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RAG_USER_TEMPLATE.format(context=context, question=question),
        },
    ]
