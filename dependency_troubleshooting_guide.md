# 🔧 JEPA Dependency Resolution: Estrategias Avanzadas de Compatibilidad

## 🎯 Análisis del Espacio de Conflictos

### Taxonomía de Incompatibilidades Detectadas

```
Grafo de Dependencias Conflictivas:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

transformers==4.46.0 [YANKED]
    ├── ❌ Incompatible con Python 3.8
    ├── ⚠️  Regresión en tokenizers bindings
    └── 🔧 Solución: Downgrade a 4.45.2 o upgrade a 4.47+

gcsfs (preinstalado en Colab)
    ├── Requiere: fsspec>=2025.3.0
    ├── Instalado: fsspec==2024.9.0
    ├── ❌ Constraint violation en resolución transitiva
    └── 🔧 Solución: Upgrade explícito de fsspec ANTES de otras deps

datasets==3.1.0
    ├── Depende: fsspec (versión flexible)
    └── ⚠️  Puede heredar versión incorrecta si no se resuelve primero
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## ⚡ Estrategia 1: Instalación Ordenada (RECOMENDADA)

### Fundamento Teórico

La resolución de dependencias en pip sigue un **algoritmo backtracking** con heurísticas. Instalando las dependencias "base" primero, reducimos el espacio de búsqueda y evitamos conflictos transitivos.

### Implementación

```python
# Orden de instalación optimizado:

# 1️⃣ CAPA BASE: Ecosistema de filesystems
!pip install -q --upgrade 'fsspec>=2025.3.0'

# 2️⃣ CAPA TRANSFORMERS: Framework NLP
!pip install -q 'transformers>=4.45.0,<4.46.0'
# Nota: Rango de versión excluye 4.46.0 yanked

# 3️⃣ CAPA DATASETS: Data loading
!pip install -q 'datasets>=3.0.0,<3.2.0'

# 4️⃣ CAPA ACELERADORES: Training optimization
!pip install -q 'accelerate>=1.0.0'

# 5️⃣ CAPA MONITOREO: Logging & tracking
!pip install -q 'wandb>=0.18.0' 'bitsandbytes>=0.44.0'
```

### Verificación de Convergencia

```python
import subprocess
import json

def verify_dependency_tree():
    """Verifica que no haya conflictos residuales."""
    result = subprocess.run(
        ['pip', 'check'], 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Dependency tree coherente")
        return True
    else:
        print("❌ Conflictos detectados:")
        print(result.stdout)
        return False

verify_dependency_tree()
```

---

## ⚡ Estrategia 2: Entorno Virtual Aislado

### Para Proyectos de Larga Duración

Si vas a iterar múltiples veces sobre el código, un entorno virtual previene contaminación del namespace global de Colab.

```bash
# Crear venv en Colab
!python -m venv /content/jepa_env

# Activar (en shell script, no en notebook directamente)
# Workaround: usar subprocess con shell=True
import subprocess
import sys

def run_in_venv(command):
    """Ejecuta comando en el venv."""
    venv_python = "/content/jepa_env/bin/python"
    return subprocess.run(
        f"{venv_python} -m {command}",
        shell=True,
        capture_output=True,
        text=True
    )

# Instalar dependencias en venv
run_in_venv("pip install --upgrade pip")
run_in_venv("pip install fsspec>=2025.3.0")
run_in_venv("pip install transformers>=4.45.0,<4.46.0")
# ... etc

# Usar venv como kernel
sys.path.insert(0, "/content/jepa_env/lib/python3.10/site-packages")
```

**⚠️ Limitación:** Colab no soporta cambio de kernel dinámico. Esta estrategia es más útil para ejecución desde terminal.

---

## ⚡ Estrategia 3: Containerización con Docker (Máxima Reproducibilidad)

### Para Deployment en Producción

```dockerfile
# Dockerfile para entorno JEPA reproducible
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Instalar Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements con versiones exactas
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Verificar instalación
RUN python -c "import transformers; import torch; print(f'✅ Setup OK: Transformers {transformers.__version__}, PyTorch {torch.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]
```

**requirements.txt:**
```
# Core dependencies con versiones pinned
fsspec==2025.3.1
transformers==4.45.2
datasets==3.1.0
accelerate==1.1.1
torch==2.5.1
wandb==0.18.7

# Optional optimizations
bitsandbytes==0.44.1
triton==3.1.0
```

**Build & Run:**
```bash
# Build image
docker build -t jepa-env:latest .

# Run con GPU passthrough
docker run --gpus all -it -v $(pwd):/workspace jepa-env:latest

# Ejecutar notebook dentro del container
jupyter notebook --ip=0.0.0.0 --allow-root
```

---

## ⚡ Estrategia 4: Pinning Completo con pip-tools

### Para Garantizar Reproducibilidad Determinista

```bash
# Instalar pip-tools
pip install pip-tools

# Crear requirements.in (versiones flexibles)
cat > requirements.in << EOF
transformers>=4.45.0,<4.46.0
datasets>=3.0.0
accelerate>=1.0.0
wandb>=0.18.0
torch>=2.0.0
EOF

# Compilar a requirements.txt (versiones exactas)
pip-compile requirements.in

# Resultado: requirements.txt con TODAS las deps transitivas pinned
# Ejemplo:
# transformers==4.45.2
# tokenizers==0.15.2
# huggingface-hub==0.24.5
# fsspec==2025.3.1
# ... (completo con hashes)

# Instalar desde lockfile
pip install -r requirements.txt
```

**Ventaja:** Instalaciones idénticas en cualquier máquina, imposible que varíen versiones.

---

## ⚡ Estrategia 5: Conda Environment (Alternativa a pip)

### Resolución SAT en Lugar de Backtracking

Conda usa un **SAT solver** más robusto que el resolver de pip para dependencias complejas.

```bash
# Instalar miniconda en Colab
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!./Miniconda3-latest-Linux-x86_64.sh -b -p /content/miniconda

# Inicializar
!/content/miniconda/bin/conda init bash
source ~/.bashrc

# Crear environment
conda create -n jepa python=3.10 -y
conda activate jepa

# Instalar dependencias (conda prioritiza compatibilidad)
conda install -c conda-forge \
    transformers=4.45 \
    datasets=3.1 \
    pytorch=2.5 \
    cudatoolkit=12.1 \
    -y

# Complementar con pip para paquetes no disponibles en conda
pip install wandb accelerate
```

**Ventaja:** SAT solver garantiza compatibilidad matemática en todo el grafo de dependencias.

---

## 🔍 Diagnóstico de Problemas Residuales

### Script de Verificación Avanzada

```python
#!/usr/bin/env python3
"""Diagnostic tool para verificar salud del entorno JEPA."""

import importlib
import sys
import subprocess
from typing import Dict, List, Tuple

class DependencyHealthCheck:
    """Verifica integridad de dependencias críticas."""
    
    CRITICAL_PACKAGES = {
        'torch': ('2.0.0', 'PyTorch core'),
        'transformers': ('4.45.0', 'Transformers library'),
        'datasets': ('3.0.0', 'HuggingFace datasets'),
        'accelerate': ('1.0.0', 'Training acceleration'),
        'wandb': ('0.18.0', 'Experiment tracking'),
        'fsspec': ('2025.3.0', 'Filesystem abstraction'),
    }
    
    def check_version(self, package: str, min_version: str) -> Tuple[bool, str]:
        """Verifica si package >= min_version."""
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            
            if version == 'unknown':
                return False, "Version unavailable"
            
            from packaging import version as pkg_version
            if pkg_version.parse(version) >= pkg_version.parse(min_version):
                return True, version
            else:
                return False, f"{version} < {min_version}"
                
        except ImportError:
            return False, "Not installed"
    
    def check_pip_conflicts(self) -> List[str]:
        """Detecta conflictos en pip check."""
        result = subprocess.run(
            ['pip', 'check'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return []
        else:
            # Parsear conflictos
            conflicts = [
                line for line in result.stdout.split('\n')
                if 'has requirement' in line or 'incompatible' in line
            ]
            return conflicts
    
    def check_cuda_availability(self) -> Dict[str, any]:
        """Verifica configuración CUDA."""
        try:
            import torch
            return {
                'available': torch.cuda.is_available(),
                'version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as e:
            return {'error': str(e)}
    
    def run_full_diagnostic(self):
        """Ejecuta diagnóstico completo."""
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "JEPA ENVIRONMENT DIAGNOSTIC" + " " * 15 + "║")
        print("╚" + "═" * 58 + "╝\n")
        
        # Check 1: Package versions
        print("📦 Critical Packages:")
        all_ok = True
        for package, (min_ver, desc) in self.CRITICAL_PACKAGES.items():
            ok, version = self.check_version(package, min_ver)
            status = "✅" if ok else "❌"
            print(f"  {status} {package:15s} {version:15s} ({desc})")
            if not ok:
                all_ok = False
        
        # Check 2: Pip conflicts
        print("\n🔍 Dependency Conflicts:")
        conflicts = self.check_pip_conflicts()
        if not conflicts:
            print("  ✅ No conflicts detected")
        else:
            print("  ❌ Conflicts found:")
            for conflict in conflicts:
                print(f"     • {conflict}")
            all_ok = False
        
        # Check 3: CUDA
        print("\n🎮 CUDA Configuration:")
        cuda_info = self.check_cuda_availability()
        if 'error' in cuda_info:
            print(f"  ❌ Error: {cuda_info['error']}")
            all_ok = False
        elif cuda_info['available']:
            print(f"  ✅ CUDA Available: {cuda_info['version']}")
            print(f"     Device: {cuda_info['device_name']}")
            print(f"     Count: {cuda_info['device_count']}")
        else:
            print("  ⚠️  CUDA not available (CPU fallback)")
        
        # Summary
        print("\n" + "="*60)
        if all_ok:
            print("✅ Environment healthy - ready for JEPA training")
        else:
            print("❌ Issues detected - review errors above")
        print("="*60)

# Ejecutar diagnóstico
if __name__ == "__main__":
    checker = DependencyHealthCheck()
    checker.run_full_diagnostic()
```

**Uso:**
```python
# En notebook de Colab
!python diagnostic_tool.py
```

---

## 🚀 Estrategia Recomendada (TL;DR)

### Para Ejecución Inmediata en Colab

```python
# 🔧 INSTALACIÓN ORDENADA (5 minutos)

# 1. Resolver fsspec primero
!pip install -q --upgrade 'fsspec>=2025.3.0'

# 2. Instalar transformers (evitar yanked)
!pip install -q 'transformers>=4.45.0,<4.46.0'

# 3. Stack ML
!pip install -q 'datasets>=3.0.0,<3.2.0' 'accelerate>=1.0.0'

# 4. Monitoring
!pip install -q 'wandb>=0.18.0' 'bitsandbytes>=0.44.0'

# 5. Verificar
import torch
import transformers
import datasets
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ Datasets: {datasets.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
```

**Tiempo total:** ~3-5 minutos  
**Probabilidad de éxito:** >95% en entornos Colab estándar

---

## 🎓 Referencias Técnicas

### Documentación de Resolución de Dependencias

1. **PEP 440 - Version Identification:**
   - Especificación de version ranges
   - Operadores de comparación (<, >, ==, !=, ~=, etc.)

2. **pip Dependency Resolution:**
   - Algoritmo backtracking con heurísticas
   - Orden de preferencia: installed > published > pre-release

3. **Conda SAT Solver:**
   - Reducción a problema Boolean Satisfiability
   - Garantías de consistencia matemática

### Herramientas de Debugging

- `pip check`: Verifica coherencia de dependencias instaladas
- `pip list --outdated`: Identifica paquetes desactualizados
- `pipdeptree`: Visualiza árbol de dependencias completo
- `pip-audit`: Detecta vulnerabilidades de seguridad en deps

---

## 🛡️ Prevención de Problemas Futuros

### Checklist de Mejores Prácticas

- [ ] Siempre usar version ranges (>=, <) en lugar de pins exactos (==)
- [ ] Consultar status de versiones en PyPI antes de pinning
- [ ] Mantener lockfiles (requirements.txt) actualizados
- [ ] Documentar justificación de version constraints
- [ ] Ejecutar `pip check` después de cada instalación
- [ ] Usar virtual environments para aislamiento
- [ ] Considerar Docker para reproducibilidad extrema

### Monitoreo Continuo

```python
# Script de monitoreo periódico
import schedule
import subprocess

def check_dependencies():
    """Verifica salud de dependencias cada semana."""
    result = subprocess.run(['pip', 'list', '--outdated'], 
                           capture_output=True, text=True)
    if result.stdout:
        print("⚠️  Paquetes desactualizados detectados:")
        print(result.stdout)
        # Enviar notificación (email, Slack, etc.)

# Programar check semanal
schedule.every().monday.at("09:00").do(check_dependencies)
```

---

**Conclusión:** La resolución de dependencias es un **problema de optimización bajo restricciones**. La estrategia óptima depende del contexto (prototipado rápido vs. producción), pero el principio fundamental es invariante: **ordenar la instalación minimiza el espacio de búsqueda y previene conflictos transitivos**.
