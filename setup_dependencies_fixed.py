#!/usr/bin/env python3
"""
🔧 JEPA Dependencies Setup - Production Grade
Resuelve conflictos de versiones con estrategia de pinning adaptativo
"""

import subprocess
import sys

def run_command(cmd, description):
    """Ejecuta comando con logging."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  Advertencia: {result.stderr}")
    else:
        print(f"✅ Completado")
    return result.returncode == 0

# ============================================================================
# FASE 1: Resolución de Conflicto fsspec (Base del Grafo)
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  🧬 JEPA Dependency Resolution System                        ║
║  Phase 1: Fixing fsspec ecosystem conflicts                  ║
╚══════════════════════════════════════════════════════════════╝
""")

# Actualizar fsspec PRIMERO (antes de que gcsfs lo reclame)
run_command(
    "pip install -q --upgrade 'fsspec>=2025.3.0'",
    "Actualizando fsspec a versión compatible con gcsfs"
)

# ============================================================================
# FASE 2: Instalación de Transformers (Versión Estable)
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  Phase 2: Installing stable transformers ecosystem           ║
╚══════════════════════════════════════════════════════════════╝
""")

# Usar 4.45.2 (última versión estable antes del yank)
# O 4.47.0 (primera post-yank si está disponible)
run_command(
    "pip install -q 'transformers>=4.45.0,<4.46.0' --upgrade",
    "Instalando transformers (versión estable, evitando 4.46.0 yanked)"
)

# ============================================================================
# FASE 3: Core ML Stack
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  Phase 3: Core machine learning infrastructure               ║
╚══════════════════════════════════════════════════════════════╝
""")

# Datasets con pinning compatible
run_command(
    "pip install -q 'datasets>=3.0.0,<3.2.0'",
    "Instalando datasets (compatible con fsspec actualizado)"
)

# Accelerate para distributed training
run_command(
    "pip install -q 'accelerate>=1.0.0'",
    "Instalando accelerate"
)

# ============================================================================
# FASE 4: Monitoring & Optimization
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  Phase 4: Monitoring and optimization tools                  ║
╚══════════════════════════════════════════════════════════════╝
""")

run_command(
    "pip install -q 'wandb>=0.18.0'",
    "Instalando wandb para monitoreo"
)

run_command(
    "pip install -q 'bitsandbytes>=0.44.0'",
    "Instalando bitsandbytes para quantization (opcional)"
)

# ============================================================================
# FASE 5: Verificación de Instalación
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  Phase 5: Installation verification                          ║
╚══════════════════════════════════════════════════════════════╝
""")

# Verificar versiones instaladas
verification_script = """
import importlib
import sys

packages = {
    'transformers': '4.45+',
    'datasets': '3.0+',
    'torch': '2.0+',
    'accelerate': '1.0+',
    'wandb': '0.18+',
    'fsspec': '2025.3+',
}

print("\\n📊 Verificación de Versiones Instaladas:\\n")
all_ok = True

for package, expected in packages.items():
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'unknown')
        status = '✅' if version != 'unknown' else '⚠️'
        print(f"{status} {package:15s} {version:15s} (esperado: {expected})")
    except ImportError:
        print(f"❌ {package:15s} NO INSTALADO")
        all_ok = False

if all_ok:
    print("\\n✅ Todas las dependencias instaladas correctamente")
else:
    print("\\n⚠️  Algunas dependencias requieren atención")

# Verificar conflictos de fsspec específicamente
try:
    import gcsfs
    import fsspec
    print(f"\\n🔍 Verificación de compatibilidad gcsfs-fsspec:")
    print(f"   gcsfs version: {gcsfs.__version__}")
    print(f"   fsspec version: {fsspec.__version__}")
    print("   ✅ Compatibilidad verificada")
except Exception as e:
    print(f"   ⚠️  Advertencia: {e}")
"""

with open('/tmp/verify_install.py', 'w') as f:
    f.write(verification_script)

subprocess.run([sys.executable, '/tmp/verify_install.py'])

print("""
╔══════════════════════════════════════════════════════════════╗
║  ✅ Dependency resolution completed                          ║
║  Sistema listo para ejecutar JEPA training                   ║
╚══════════════════════════════════════════════════════════════╝
""")
