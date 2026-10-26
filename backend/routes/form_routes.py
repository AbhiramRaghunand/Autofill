import base64
import logging
from pathlib import Path

from bson import ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.auth import get_authenticated_user_id
from backend.security.audit_logger import log_security_event
from backend.services.form_filling_service import FormFillingService
from backend.utils.secure_files import secure_delete_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/forms", tags=["Forms"])
form_filling_service = FormFillingService()


@router.post("/fill")
async def fill_form(
    request: Request,
    form: UploadFile = File(...),
    user_id: ObjectId = Depends(get_authenticated_user_id),
):
    client_ip = request.client.host if request.client else None

    try:
        result = await form_filling_service.fill(
            uploaded_file=form,
            user_id=user_id,
        )

        output_path = Path(result["output_path"])
        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="The completed form could not be found.",
            )

        output_suffix = output_path.suffix.lower()
        media_type = "application/pdf" if output_suffix == ".pdf" else "image/png"

        try:
            file_bytes = output_path.read_bytes()
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Unable to read the completed form.",
            ) from exc
        finally:
            # Ephemeral file lifecycle: purge generated form from disk immediately after encoding.
            secure_delete_file(output_path)

        log_security_event(
            action="FORM_AUTOFILL",
            user_id=user_id,
            details={"review_items_count": len(result.get("review_items", []))},
            ip_address=client_ip,
            status="SUCCESS",
        )

        return JSONResponse(
            content={
                "filename": f"filled_form{output_suffix}",
                "media_type": media_type,
                "file_base64": base64.b64encode(file_bytes).decode("ascii"),
                "review_items": result.get("review_items", []),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Keep the detailed traceback in server logs, not in the client response.
        logger.exception("Unexpected error while processing /api/forms/fill")

        log_security_event(
            action="FORM_AUTOFILL_FAILED",
            user_id=user_id,
            details={"error_type": type(exc).__name__},
            ip_address=client_ip,
            status="FAILURE",
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to process the form. Please try again.",
        ) from exc
