"""
🛡️ JEPA TRAINING LOOP - FAULT TOLERANT VERSION
Resolución de crashes mediante:
- Memory monitoring con early warning
- Adaptive batch sizing según VRAM
- Timeout handling en streaming
- Granular logging para debug
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import wandb
import time
from typing import Optional
import psutil
import gc

class ResilientJEPATrainer:
    """Trainer con fault tolerance y diagnóstico avanzado."""
    
    def __init__(
        self,
        model,
        criterion,
        train_dataloader,
        num_steps: int = 10000,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 4,
        gradient_clip_val: float = 1.0,
        log_interval: int = 100,
        save_interval: int = 2000,
        save_path: str = "./checkpoints",
        max_batch_size: int = 8,
        enable_memory_monitoring: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.num_steps = num_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_path = save_path
        self.enable_memory_monitoring = enable_memory_monitoring
        
        # Adaptive batch sizing
        self.current_batch_size = self._determine_optimal_batch_size(max_batch_size)
        
        # Optimizer con LR diferenciado
        self.optimizer = torch.optim.AdamW([
            {'params': model.student.parameters(), 'lr': learning_rate},
            {'params': model.predictor.parameters(), 'lr': learning_rate * 10},
        ], betas=(0.9, 0.999), weight_decay=0.01)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_steps,
            eta_min=learning_rate * 0.01,
        )
        
        # Scaler para mixed precision (API moderna)
        from torch.amp import GradScaler
        self.scaler = GradScaler('cuda')
        
        # Estadísticas de monitoreo
        self.stats = {
            'oom_count': 0,
            'timeout_count': 0,
            'nan_loss_count': 0,
        }
    
    def _determine_optimal_batch_size(self, max_size: int) -> int:
        """Determina batch size óptimo basado en VRAM disponible."""
        if not torch.cuda.is_available():
            return max_size
        
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        available = total_vram - allocated - 2.0  # Reserve 2GB buffer
        
        # Heurística: ~1GB por batch de 8 ejemplos
        optimal_size = int((available / 1.0) * 8)
        optimal_size = min(optimal_size, max_size)
        optimal_size = max(optimal_size, 2)  # Mínimo 2
        
        print(f"🔍 VRAM Analysis:")
        print(f"   Total: {total_vram:.1f}GB")
        print(f"   Allocated: {allocated:.1f}GB")
        print(f"   Available: {available:.1f}GB")
        print(f"   ✅ Optimal batch size: {optimal_size}")
        
        return optimal_size
    
    def _monitor_memory(self):
        """Monitorea uso de memoria GPU."""
        if not self.enable_memory_monitoring:
            return {}
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
        
        # Warning si se acerca al límite
        if allocated > 12.0:  # 12GB de 15GB
            print(f"⚠️  High VRAM usage: {allocated:.1f}GB / 15GB")
        
        return {
            'memory/allocated_gb': allocated,
            'memory/reserved_gb': reserved,
            'memory/max_allocated_gb': max_allocated,
        }
    
    def _safe_next_batch(self, dataloader_iter, timeout_sec: int = 60):
        """Obtiene siguiente batch con timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Dataloader timeout")
        
        # Configurar timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        
        try:
            batch = next(dataloader_iter)
            signal.alarm(0)  # Cancelar timeout
            return batch
        except TimeoutError:
            signal.alarm(0)
            self.stats['timeout_count'] += 1
            print(f"⏱️  Dataloader timeout (attempt {self.stats['timeout_count']})")
            # Reiniciar iterador
            return None
        except StopIteration:
            signal.alarm(0)
            return None
    
    def _handle_oom(self):
        """Maneja Out-of-Memory errors."""
        self.stats['oom_count'] += 1
        print(f"💀 OOM detected (count: {self.stats['oom_count']})")
        
        # Limpieza agresiva
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reducir batch size
        if self.current_batch_size > 2:
            self.current_batch_size = max(2, self.current_batch_size // 2)
            print(f"   📉 Reducing batch size to {self.current_batch_size}")
        else:
            print("   ❌ Cannot reduce batch size further - aborting")
            raise RuntimeError("Persistent OOM with minimum batch size")
    
    def train(self):
        """Training loop con fault tolerance."""
        from pathlib import Path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        self.model.to('cuda')
        self.model.train()
        
        print(f"\n{'='*60}")
        print(f"🚀 FAULT-TOLERANT JEPA TRAINING")
        print(f"{'='*60}")
        print(f"Total steps: {self.num_steps}")
        print(f"Batch size (adaptive): {self.current_batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.current_batch_size * self.gradient_accumulation_steps}")
        print(f"Memory monitoring: {'Enabled' if self.enable_memory_monitoring else 'Disabled'}")
        print(f"{'='*60}\n")
        
        global_step = 0
        self.optimizer.zero_grad()
        
        # Inicializar dataloader con retry logic
        print("🔄 Initializing dataloader...")
        dataloader_iter = None
        for attempt in range(3):
            try:
                dataloader_iter = iter(self.train_dataloader)
                print(f"✅ Dataloader initialized (attempt {attempt+1})")
                break
            except Exception as e:
                print(f"⚠️  Dataloader init failed (attempt {attempt+1}): {e}")
                time.sleep(5)
        
        if dataloader_iter is None:
            raise RuntimeError("Failed to initialize dataloader after 3 attempts")
        
        # Forzar primer batch para detectar problemas early
        print("🔍 Fetching first batch (this may take 5-15 minutes)...")
        print("   Please wait while streaming buffer fills...")
        
        start_time = time.time()
        first_batch = None
        
        while first_batch is None:
            first_batch = self._safe_next_batch(dataloader_iter, timeout_sec=300)
            if first_batch is None:
                print("   ⏳ Retrying... (streaming dataset initialization)")
                dataloader_iter = iter(self.train_dataloader)
                time.sleep(10)
            
            # Timeout total: 30 minutos
            if time.time() - start_time > 1800:
                raise RuntimeError("First batch timeout after 30 minutes")
        
        elapsed = time.time() - start_time
        print(f"✅ First batch received in {elapsed/60:.1f} minutes")
        print(f"   Batch keys: {list(first_batch.keys())}")
        print(f"   Context shape: {first_batch['context_ids'].shape}")
        print(f"   Target shape: {first_batch['target_ids'].shape}")
        
        # Training loop principal
        print(f"\n{'='*60}")
        print(f"🎯 Starting training iterations")
        print(f"{'='*60}\n")
        
        iteration_times = []
        
        while global_step < self.num_steps:
            step_start = time.time()
            
            try:
                # Gradient accumulation loop
                for accum_step in range(self.gradient_accumulation_steps):
                    # Obtener batch
                    if accum_step == 0 and global_step == 0:
                        batch = first_batch  # Usar primer batch ya cargado
                    else:
                        batch = self._safe_next_batch(dataloader_iter, timeout_sec=120)
                        
                        if batch is None:
                            print("⚠️  Dataloader exhausted, reinitializing...")
                            dataloader_iter = iter(self.train_dataloader)
                            batch = self._safe_next_batch(dataloader_iter)
                    
                    # Mover a GPU
                    batch = {k: v.to('cuda') for k, v in batch.items()}
                    
                    # Forward pass con autocast
                    with autocast(device_type='cuda'):
                        z_pred, z_target, teacher_logits = self.model(
                            context_ids=batch['context_ids'],
                            context_mask=batch['context_mask'],
                            target_ids=batch['target_ids'],
                            target_mask=batch['target_mask'],
                        )
                        
                        # Calcular pérdida
                        loss, metrics = self.criterion(
                            z_pred=z_pred,
                            z_target=z_target,
                            teacher_logits=teacher_logits,
                            target_ids=batch['target_ids'],
                        )
                        
                        # Verificar NaN
                        if torch.isnan(loss):
                            self.stats['nan_loss_count'] += 1
                            print(f"⚠️  NaN loss detected (count: {self.stats['nan_loss_count']})")
                            if self.stats['nan_loss_count'] > 10:
                                raise RuntimeError("Persistent NaN losses - training unstable")
                            continue
                        
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Backward
                    self.scaler.scale(loss).backward()
                
                # Optimizer step
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_val,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update Teacher
                self.model.update_teacher_ema()
                self.model.update_ema_momentum(global_step, self.num_steps)
                
                # Scheduler
                self.scheduler.step()
                
                global_step += 1
                
                # Logging
                step_time = time.time() - step_start
                iteration_times.append(step_time)
                
                if global_step % self.log_interval == 0:
                    # Métricas de entrenamiento
                    log_metrics = {
                        **metrics,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/ema_momentum': self.model.ema_momentum.item(),
                        'train/grad_norm': grad_norm.item(),
                        'train/step_time': step_time,
                        'train/steps_per_sec': 1.0 / (sum(iteration_times[-10:]) / min(10, len(iteration_times))),
                        **self._monitor_memory(),
                    }
                    
                    wandb.log(log_metrics, step=global_step)
                    
                    avg_time = sum(iteration_times[-10:]) / min(10, len(iteration_times))
                    eta_hours = (self.num_steps - global_step) * avg_time / 3600
                    
                    print(f"Step {global_step:5d}/{self.num_steps} | "
                          f"Loss: {metrics['loss/total']:.4f} | "
                          f"Inv: {metrics['loss/invariance']:.4f} | "
                          f"Var: {metrics['loss/variance']:.4f} | "
                          f"Cov: {metrics['loss/covariance']:.4f} | "
                          f"Time: {step_time:.2f}s | "
                          f"ETA: {eta_hours:.1f}h")
                
                # Checkpointing
                if global_step % self.save_interval == 0:
                    checkpoint_path = f"{self.save_path}/checkpoint_step_{global_step}.pt"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'metrics': metrics,
                        'stats': self.stats,
                    }, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._handle_oom()
                else:
                    raise
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total steps: {global_step}")
        print(f"OOM events: {self.stats['oom_count']}")
        print(f"Timeouts: {self.stats['timeout_count']}")
        print(f"NaN losses: {self.stats['nan_loss_count']}")
        print(f"{'='*60}\n")


# ============================================================================
# USO: Reemplazar la función train_jepa() por esta clase
# ============================================================================

"""
# En lugar de:
train_jepa(model, criterion, train_dataloader, ...)

# Usar:
trainer = ResilientJEPATrainer(
    model=model,
    criterion=criterion,
    train_dataloader=train_dataloader,
    num_steps=10000,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    max_batch_size=8,  # Se ajustará automáticamente
    enable_memory_monitoring=True,
)

trainer.train()
"""
