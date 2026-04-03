import os

import torch


class CheckpointManager:
    def __init__(self):
        self._best_metric: float = float("inf")

    def save(self, trainer, path: str, epoch: int, loss: float) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "epoch": epoch,
                "loss": loss,
            },
            path,
        )

    def load(self, trainer, path: str) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=trainer.config.device, weights_only=False)
        trainer.model.load_state_dict(state["model_state_dict"])
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
        return state["epoch"]

    def save_best(self, trainer, save_dir: str, metric: float, epoch: int) -> bool:
        if metric < self._best_metric:
            self._best_metric = metric
            best_path = os.path.join(save_dir, "best_model.pt")
            self.save(trainer, best_path, epoch=epoch, loss=metric)
            return True
        return False

    def list_checkpoints(self, save_dir: str) -> list:
        if not os.path.isdir(save_dir):
            return []
        files = [
            f for f in os.listdir(save_dir)
            if f.startswith("epoch_") and f.endswith(".pt")
        ]
        files.sort(key=lambda f: int(f.replace("epoch_", "").replace(".pt", "")))
        return [os.path.join(save_dir, f) for f in files]
