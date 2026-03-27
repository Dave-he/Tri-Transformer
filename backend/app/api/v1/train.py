import json
import threading
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.config import settings
from app.core.database import get_db
from app.dependencies import get_current_user
from app.model.trainer import TriTransformerTrainer, TrainerConfig
from app.model.tri_transformer import TriTransformerConfig
from app.models.train_job import TrainJob
from app.models.user import User
from app.schemas.train import TrainJobRequest, TrainJobResponse
from app.services.train.train_service import TrainService

router = APIRouter()

_cancel_events: dict[str, threading.Event] = {}


def _job_to_response(job) -> TrainJobResponse:
    config = json.loads(job.config) if job.config else {}
    return TrainJobResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        config=config,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _run_training(job_id: str, job_type: str, user_config: dict, db_url: str):
    """在后台线程中同步执行 PyTorch 训练，完成后更新 DB 状态。"""
    import asyncio

    cancel_event = _cancel_events.setdefault(job_id, threading.Event())

    model_cfg = TriTransformerConfig(
        vocab_size=user_config.get("vocab_size", settings.train_vocab_size),
        d_model=user_config.get("d_model", settings.train_d_model),
        num_heads=user_config.get("num_heads", settings.train_num_heads),
        num_layers_i=user_config.get("num_layers", settings.train_num_layers),
        num_layers_c=max(2, user_config.get("num_layers", settings.train_num_layers) // 2),
        num_layers_o=user_config.get("num_layers", settings.train_num_layers),
        max_len=user_config.get("max_seq_len", settings.train_max_seq_len),
    )
    trainer_cfg = TrainerConfig(
        job_type=job_type,
        num_epochs=user_config.get("num_epochs", settings.train_epochs_default),
        learning_rate=user_config.get("learning_rate", settings.train_lr_default),
        vocab_size=model_cfg.vocab_size,
        seq_len=user_config.get("seq_len", 32),
        device=settings.train_device,
        model_config=model_cfg,
    )
    metrics_history: list[dict] = []

    def _on_epoch(m: dict):
        metrics_history.append(m)

    async def _update_db(status: str, metrics: list[dict]):
        engine = create_async_engine(db_url)
        sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with sm() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(TrainJob).where(TrainJob.id == job_id)
                )
                job = result.scalar_one_or_none()
                if job:
                    cfg = json.loads(job.config) if job.config else {}
                    cfg["metrics"] = metrics
                    job.config = json.dumps(cfg)
                    job.status = status
                    job.updated_at = datetime.now(timezone.utc)
                    await session.commit()
        except Exception:
            pass
        finally:
            try:
                await engine.dispose()
            except Exception:
                pass

    try:
        asyncio.run(_update_db("running", []))
        trainer = TriTransformerTrainer(
            config=trainer_cfg,
            metrics_callback=_on_epoch,
            cancel_event=cancel_event,
        )
        trainer.train()
        final_status = "cancelled" if cancel_event.is_set() else "completed"
        asyncio.run(_update_db(final_status, metrics_history))
    except Exception as exc:
        asyncio.run(_update_db("failed", metrics_history))
    finally:
        _cancel_events.pop(job_id, None)


@router.post("/jobs", response_model=TrainJobResponse, status_code=202)
async def submit_job(
    payload: TrainJobRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = TrainService(db)
    job = await svc.submit_job(
        user_id=current_user.id,
        job_type=payload.job_type,
        config=payload.config,
    )
    background_tasks.add_task(
        _run_training,
        job_id=job.id,
        job_type=job.job_type,
        user_config=payload.config,
        db_url=settings.database_url,
    )
    return _job_to_response(job)


@router.get("/jobs", response_model=list[TrainJobResponse])
async def list_jobs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = TrainService(db)
    jobs = await svc.list_jobs(user_id=current_user.id)
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=TrainJobResponse)
async def get_job(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = TrainService(db)
    job = await svc.get_job(job_id=job_id, user_id=current_user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.delete("/jobs/{job_id}", status_code=200)
async def cancel_job(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    event = _cancel_events.get(job_id)
    if event:
        event.set()

    svc = TrainService(db)
    job = await svc.cancel_job(job_id=job_id, user_id=current_user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status}
