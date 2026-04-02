import asyncio

from app.database.session import async_session_factory
from app.services.workflow_service import WorkflowService
from app.workers.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3, autoretry_for=(Exception,), retry_backoff=True, retry_jitter=True)
def process_candidate_workflow_task(self, candidate_id: int, company_id: int):
    async def _run():
        async with async_session_factory() as session:
            await WorkflowService().run_candidate_workflow(session, candidate_id, company_id)
            await session.commit()

    return asyncio.run(_run())

