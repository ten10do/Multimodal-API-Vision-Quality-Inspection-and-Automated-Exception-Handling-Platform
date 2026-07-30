import asyncio
from uuid import UUID

from celery import Celery

from app.config import get_settings
from app.workflow import process_inspection

settings = get_settings()
celery_app = Celery("vision_qc", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_always_eager=settings.celery_task_always_eager,
)


@celery_app.task(
    name="vision_qc.process_inspection",
    autoretry_for=(OSError,),
    retry_backoff=True,
    max_retries=3,
)
def process_inspection_task(inspection_id: str) -> None:
    asyncio.run(process_inspection(UUID(inspection_id)))
