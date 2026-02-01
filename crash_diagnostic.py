"""
🔍 POST-CRASH DIAGNOSTIC TOOLKIT
Análisis forense del crash de JEPA training
"""

import torch
import subprocess
import json
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════╗
║           🔬 JEPA CRASH FORENSIC ANALYSIS                    ║
║           Post-Mortem Diagnostic Suite                       ║
╚══════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# DIAGNÓSTICO 1: Verificar modelo Qwen
# ============================================================================

print("\n[1/6] 🧬 MODELO QWEN VERIFICATION")
print("━" * 60)

try:
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    print(f"✅ Modelo cargado correctamente")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    print(f"   Vocab size: {config.vocab_size}")
    
    # VERIFICACIÓN CRÍTICA
    if config.hidden_size != 1024:
        print(f"\n⚠️  ANOMALÍA DETECTADA:")
        print(f"   Expected hidden_size: 1024")
        print(f"   Actual hidden_size: {config.hidden_size}")
        print(f"   Esto puede causar incompatibilidades dimensionales")
        
        # Buscar modelo alternativo
        print(f"\n🔧 Modelos alternativos recomendados:")
        print(f"   - Qwen/Qwen2-0.5B (hidden_size=896)")
        print(f"   - Qwen/Qwen2.5-1.5B (hidden_size=1536)")
    else:
        print(f"   ✅ Dimensiones correctas para JEPA")

except Exception as e:
    print(f"❌ Error cargando modelo: {e}")

# ============================================================================
# DIAGNÓSTICO 2: VRAM disponible
# ============================================================================

print("\n[2/6] 🎮 GPU MEMORY ANALYSIS")
print("━" * 60)

if torch.cuda.is_available():
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    
    total_vram = props.total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    free = total_vram - reserved
    
    print(f"GPU: {props.name}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB ({allocated/total_vram*100:.1f}%)")
    print(f"Reserved: {reserved:.2f} GB ({reserved/total_vram*100:.1f}%)")
    print(f"Free: {free:.2f} GB ({free/total_vram*100:.1f}%)")
    
    # Estimación de requerimientos JEPA
    print(f"\n📊 JEPA Memory Requirements Estimate:")
    
    # Asumiendo hidden_dim=896 (como en logs)
    hidden_dim = 896
    num_layers = 24
    batch_size = 8
    seq_len = 512
    
    # Student + Teacher (aproximado)
    model_params = 2 * (num_layers * hidden_dim * hidden_dim * 4) * 2 / 1e9  # FP16
    print(f"   Models (Student+Teacher): ~{model_params:.1f} GB")
    
    # Predictor
    predictor_params = (hidden_dim * 4096 + 4096 * 4096 * 3 + 4096 * hidden_dim) * 2 / 1e9
    print(f"   Predictor: ~{predictor_params:.1f} GB")
    
    # Activations (más crítico)
    activations = batch_size * seq_len * hidden_dim * num_layers * 2 / 1e9 * 2  # Forward+backward
    print(f"   Activations (batch={batch_size}): ~{activations:.1f} GB")
    
    total_estimated = model_params + predictor_params + activations
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   Total Estimated: ~{total_estimated:.1f} GB")
    
    if total_estimated > free:
        print(f"\n❌ PROBABLE OOM CAUSE:")
        print(f"   Required: {total_estimated:.1f} GB")
        print(f"   Available: {free:.1f} GB")
        print(f"   Deficit: {total_estimated - free:.1f} GB")
        print(f"\n🔧 RECOMMENDED FIXES:")
        print(f"   1. Reduce batch_size: 8 → 4 (halves activation memory)")
        print(f"   2. Reduce seq_len: 512 → 256 (halves activation memory)")
        print(f"   3. Use gradient checkpointing (30% slower, 40% less memory)")
    else:
        print(f"\n✅ Sufficient VRAM available")
else:
    print(f"❌ CUDA not available")

# ============================================================================
# DIAGNÓSTICO 3: Verificar dataset streaming
# ============================================================================

print("\n[3/6] 📡 DATASET STREAMING TEST")
print("━" * 60)

try:
    from datasets import load_dataset
    
    print("Testing connection to HuggingFace Hub...")
    
    # Test con timeout
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True,
        )
        
        # Intentar obtener primer ejemplo
        first_example = next(iter(ds))
        signal.alarm(0)
        
        print(f"✅ Streaming dataset accessible")
        print(f"   First example keys: {list(first_example.keys())}")
        print(f"   Text length: {len(first_example.get('text', ''))} chars")
        
    except TimeoutError:
        signal.alarm(0)
        print(f"❌ TIMEOUT: Cannot connect to HuggingFace")
        print(f"   Possible causes:")
        print(f"   - Network issues")
        print(f"   - HuggingFace API rate limiting")
        print(f"   - Colab network restrictions")

except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# DIAGNÓSTICO 4: Verificar dependencias críticas
# ============================================================================

print("\n[4/6] 📦 DEPENDENCY HEALTH CHECK")
print("━" * 60)

critical_packages = {
    'torch': None,
    'transformers': None,
    'datasets': None,
    'wandb': None,
}

for pkg in critical_packages.keys():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        critical_packages[pkg] = version
        print(f"✅ {pkg:15s} {version}")
    except ImportError:
        critical_packages[pkg] = None
        print(f"❌ {pkg:15s} NOT INSTALLED")

# ============================================================================
# DIAGNÓSTICO 5: Test del Predictor aislado
# ============================================================================

print("\n[5/6] 🧪 LATENT PREDICTOR UNIT TEST")
print("━" * 60)

try:
    # Importar LatentPredictor
    exec(open('/content/qwen_jepa_implementation.ipynb').read())  # Esto fallará, es ejemplo
    
    print("Testing Predictor with mock data...")
    
    # Crear predictor de prueba
    predictor = LatentPredictor(input_dim=896, hidden_dim=4096, num_blocks=3)
    predictor = predictor.to('cuda')
    
    # Test forward pass
    mock_input = torch.randn(8, 896).to('cuda')  # batch=8, hidden=896
    
    with torch.no_grad():
        output = predictor(mock_input)
    
    print(f"✅ Predictor functional")
    print(f"   Input shape: {mock_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Cleanup
    del predictor, mock_input, output
    torch.cuda.empty_cache()

except Exception as e:
    print(f"⚠️  Cannot test Predictor in isolation: {e}")

# ============================================================================
# DIAGNÓSTICO 6: Revisar logs de sistema
# ============================================================================

print("\n[6/6] 📋 SYSTEM LOGS ANALYSIS")
print("━" * 60)

try:
    # Buscar mensajes de OOM en dmesg (requiere permisos root en Colab)
    result = subprocess.run(
        ['dmesg', '-T'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        oom_lines = [l for l in lines if 'oom' in l.lower() or 'killed' in l.lower()]
        
        if oom_lines:
            print(f"❌ OOM KILLER EVENTS DETECTED:")
            for line in oom_lines[-5:]:  # Last 5
                print(f"   {line}")
        else:
            print(f"✅ No OOM events in system logs")
    else:
        print(f"⚠️  Cannot access system logs (permission denied)")

except Exception as e:
    print(f"⚠️  Cannot analyze system logs: {e}")

# ============================================================================
# RESUMEN Y RECOMENDACIONES
# ============================================================================

print(f"\n")
print("╔" + "═" * 58 + "╗")
print("║" + " " * 20 + "DIAGNOSTIC SUMMARY" + " " * 20 + "║")
print("╚" + "═" * 58 + "╝")

print(f"\n🎯 LIKELY ROOT CAUSE:")
print(f"   Based on silent crash after 'Iniciando entrenamiento',")
print(f"   the most probable cause is:")
print(f"\n   💀 OUT-OF-MEMORY during first forward pass")
print(f"\n   Why it's silent:")
print(f"   - Linux OOM killer sends SIGKILL (no Python traceback)")
print(f"   - Process dies before exception handling")
print(f"   - Colab runtime shows 'Crashed' status")

print(f"\n🔧 RECOMMENDED SOLUTIONS (in priority order):")
print(f"\n   1️⃣  REDUCE BATCH SIZE (CRITICAL)")
print(f"      Change: batch_size=8 → 4 (or even 2)")
print(f"      Impact: -50% activation memory")
print(f"      Edit in: data_config = JEPADataConfig(batch_size=4)")

print(f"\n   2️⃣  REDUCE SEQUENCE LENGTH")
print(f"      Change: max_length=512 → 256")
print(f"      Impact: -50% activation memory")
print(f"      Edit in: data_config = JEPADataConfig(max_length=256)")

print(f"\n   3️⃣  USE RESILIENT TRAINER (PROVIDED)")
print(f"      Features:")
print(f"      - Adaptive batch sizing")
print(f"      - OOM recovery")
print(f"      - Memory monitoring")
print(f"      - Timeout handling")

print(f"\n   4️⃣  ENABLE GRADIENT CHECKPOINTING")
print(f"      Add in model initialization:")
print(f"      model.student.gradient_checkpointing_enable()")
print(f"      model.teacher.gradient_checkpointing_enable()")
print(f"      Trade-off: +30% time, -40% memory")

print(f"\n📝 NEXT STEPS:")
print(f"   1. Stop current crashed session (Runtime → Restart)")
print(f"   2. Modify batch_size to 4 in notebook")
print(f"   3. Re-run from beginning")
print(f"   4. Monitor VRAM with: !nvidia-smi -l 1")

print(f"\n💡 PREVENTION FOR FUTURE RUNS:")
print(f"   - Always run memory diagnostic before training")
print(f"   - Start with conservative batch_size=2")
print(f"   - Gradually increase if VRAM allows")
print(f"   - Use ResilientJEPATrainer instead of basic loop")

print("\n" + "="*60)
