from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from app.schemas import ErrorResponse, IngestResponse

router = APIRouter()

_ingest_service = None


def set_ingest_service(service):
    global _ingest_service
    _ingest_service = service


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ingest(
    file: UploadFile | None = File(default=None),
    path: str | None = Form(default=None),
):
    if file and path:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "只能传 file 或 path，不能同时传",
                "status": 400,
            },
        )

    if not file and not path:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "必须传 file 或 path",
                "status": 400,
            },
        )

    try:
        if file:
            content = await file.read()
            text = content.decode("utf-8")
            filename = file.filename or "uploaded"
            result = _ingest_service.ingest_text(text, filename=filename)
        else:
            result = _ingest_service.ingest_file(path)

        return IngestResponse(**result)

    except ValueError as e:
        error_code = str(e)
        status_map = {
            "unsupported_file_type": 400,
            "file_not_found": 400,
            "empty_document": 400,
        }
        message_map = {
            "unsupported_file_type": "只支持 .md / .txt 文件",
            "file_not_found": f"文件不存在: {path}",
            "empty_document": "文件内容为空",
        }
        raise HTTPException(
            status_code=status_map.get(error_code, 500),
            detail={
                "error": error_code,
                "message": message_map.get(error_code, str(e)),
                "status": status_map.get(error_code, 500),
            },
        )
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": str(e), "status": 500},
        )
