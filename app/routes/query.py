import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.schemas import ErrorResponse, QueryRequest, QueryResponse

router = APIRouter()

_query_service = None


def set_query_service(service):
    global _query_service
    _query_service = service


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
)
async def query(req: QueryRequest):
    try:
        if req.stream:
            return StreamingResponse(
                _stream_response(req.question, req.top_k),
                media_type="text/event-stream",
            )

        result = await _query_service.query_sync(req.question, top_k=req.top_k)
        return QueryResponse(**result)

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "llm_error", "message": str(e), "status": 500},
        )


async def _stream_response(question: str, top_k: int):
    try:
        async for event in _query_service.query_stream(question, top_k=top_k):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.exception("Stream query failed")
        yield f"data: {json.dumps({'error': 'llm_error', 'message': str(e)}, ensure_ascii=False)}\n\n"
