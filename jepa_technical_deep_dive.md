# 🧬 JEPA Architecture Deep Dive: Mathematical Foundations & Production Considerations

## 📐 Análisis Matemático del Espacio de Representación

### 1. Geometría del Espacio Latente

La arquitectura JEPA induce una **variedad diferencial** en el espacio de embeddings ℝ^d donde las propiedades geométricas están controladas por VICReg:

#### 1.1 Invariant Manifold

La pérdida de invarianza L_inv fuerza la proyección de contexto y target al mismo punto en el manifold latente:

```
π: X → Z
donde π(x) ≈ π(y) para pares semánticamente relacionados (x,y)
```

**Propiedad crítica:** Esta es una contracción métrica con factor λ, no una isometría. El espacio latente "comprime" información irrelevante mientras preserva señal semántica.

#### 1.2 Variance Preservation

La pérdida L_var garantiza que el volumen del manifold no colapse:

```
det(Cov(Z)) ≥ γ^d
```

Esto previene la degeneración dimensional (mode collapse) que plaga a los autoencoders variacionales sin regularización apropiada.

#### 1.3 Decorrelation Constraint

La pérdida L_cov fuerza ortogonalización de características:

```
Cov(Z) ≈ σ²I + ε·E
donde E es ruido residual pequeño
```

Resultado: El espacio latente aprende un **sistema de coordenadas canónico** donde cada dimensión captura un factor de variación independiente.

---

## 🔬 Comparativa: JEPA vs Paradigmas Tradicionales

### Tabla Comparativa de Eficiencia Computacional

| Arquitectura | Tokens/Forward | Operaciones FLOPs | Memoria (Activations) | Colapso Dimensional |
|--------------|----------------|-------------------|-----------------------|---------------------|
| **Transformer Autoregresivo** | T × T (cuadrático) | O(T² · d) | O(T² · d) | No aplica |
| **BERT (MLM)** | T (paralelo) | O(T · d²) | O(T · d) | Moderado (sin VICReg) |
| **SimCLR** | 1 (embedding único) | O(T · d²) | O(d) | **Alto** (requiere grandes batches) |
| **JEPA + VICReg** | 1 (embedding único) | O(T · d²) | O(d) | **Muy bajo** (regularizado) |

**Ventaja competitiva de JEPA:**
- Inferencia en espacio latente (no token-a-token)
- Regularización explícita contra colapso
- Memoria de activación constante respecto a longitud de secuencia

---

## 🎯 SIGReg: Teoría de Información Aplicada

### Fundamento Matemático

SIGReg pondera la pérdida por el contenido informativo del target:

```
I(y) = -log P(y | Context_LM)
```

Donde P viene de un modelo de lenguaje base (en nuestro caso, las proyecciones del Teacher).

**Interpretación:**
- Tokens predecibles (artículos, preposiciones): Bajo peso
- Tokens densos en información (términos técnicos, entidades): Alto peso

### Implementación Numérica Estable

```python
# Cálculo robusto de self-information
log_probs = F.log_softmax(logits, dim=-1)  # Estable numéricamente
self_info = -log_probs.gather(-1, target_ids.unsqueeze(-1))

# Normalización con sigmoid (evita explosión)
weights = alpha + (1-alpha) * torch.sigmoid(self_info.mean(dim=1))
```

**Detalle crítico:** Usamos sigmoid para acotar los pesos en [α, 1], evitando que ejemplos con información extremadamente alta desestabilicen el gradiente.

---

## ⚙️ Ingeniería de Producción: Consideraciones Críticas

### 1. Gestión de Memoria en Entrenamiento

#### Problema: OOM en Batches Grandes

VICReg requiere batches grandes para estimar matrices de covarianza. Soluciones:

**a) Gradient Accumulation (implementado)**
```python
effective_batch = batch_size × accumulation_steps
# Costo: latencia aumentada
```

**b) Gradient Checkpointing (para Qwen > 1B)**
```python
from torch.utils.checkpoint import checkpoint

outputs = checkpoint(model.student, input_ids, attention_mask)
# Trade-off: 30% más tiempo, -40% memoria
```

**c) Covariance Estimation con Moving Statistics**
```python
# Mantener running covariance en lugar de calcular en batch
self.register_buffer('running_cov', torch.zeros(d, d))
```

### 2. Estabilidad Numérica en VICReg

#### Problema: División por Cero en Variance Loss

```python
# ❌ Implementación ingenua
std = torch.sqrt(z.var(dim=0))  # Puede ser exactamente 0

# ✅ Implementación robusta
eps = 1e-4
std = torch.sqrt(z.var(dim=0) + eps)
```

#### Problema: Explosión de Covarianza en Dimensiones Altas

Para d=1024, la matriz de covarianza tiene ~1M elementos. Solución:

```python
# Normalización de covarianza antes del cálculo
z_normalized = (z - z.mean(0)) / (z.std(0) + eps)
cov = z_normalized.T @ z_normalized / (batch_size - 1)
```

### 3. EMA Schedule Coseno: Derivación

La actualización coseno del Teacher estabiliza convergencia:

```
m(t) = 1 - (1 - m_base) · [1 + cos(πt/T)] / 2
```

**Propiedades deseables:**
- t=0: m ≈ m_base (Teacher cambia lentamente al inicio)
- t=T/2: m → 1 (Teacher casi congelado)
- t=T: m = 1 (Teacher completamente congelado)

**Justificación teórica:** 
En las etapas finales, queremos que el Teacher sea un "anchor" estable para que el Student refine su alineación sin perseguir un target móvil.

---

## 🔍 Diagnóstico de Patologías del Entrenamiento

### Symptom 1: Variance Collapse

**Síntoma:** `metrics/z_pred_std` → 0

**Diagnóstico:**
```python
# Verificar gradientes de la pérdida de varianza
print(f"Grad Var Loss: {loss.grad_fn}")
# Si es None, el término no está propagando
```

**Soluciones:**
1. Aumentar μ (peso de variance): 25 → 50
2. Reducir learning rate del Predictor
3. Verificar BatchNorm (debe estar en modo train)

### Symptom 2: Teacher-Student Divergence

**Síntoma:** `loss/invariance` no converge

**Diagnóstico:**
```python
# Calcular distancia Wasserstein entre distribuciones
from scipy.stats import wasserstein_distance
w_dist = wasserstein_distance(
    z_pred.flatten().cpu().numpy(),
    z_target.flatten().cpu().numpy()
)
```

**Soluciones:**
1. Reducir EMA momentum: 0.996 → 0.99
2. Aumentar λ (peso de invariance): 25 → 40
3. Verificar que stop_gradient está activo en Teacher

### Symptom 3: Covariance Explosion

**Síntoma:** `loss/covariance` >> 1.0

**Diagnóstico:**
```python
# Eigenvalues de matriz de covarianza
eigenvalues = torch.linalg.eigvalsh(cov_matrix)
print(f"Condition number: {eigenvalues.max() / eigenvalues.min()}")
```

**Soluciones:**
1. Gradient clipping más agresivo: 1.0 → 0.5
2. Reducir ν (peso de covariance): 1.0 → 0.5
3. Normalización espectral en el Predictor

---

## 🚀 Optimizaciones Avanzadas

### 1. Predictor con Attention Pooling

En lugar de mean pooling, usar self-attention:

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch, seq, hidden]
        attn_weights = self.attention(hidden_states)  # [batch, seq, 1]
        attn_weights = attn_weights.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 
            float('-inf')
        )
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (hidden_states * attn_weights).sum(dim=1)
        return pooled
```

**Ventaja:** Aprende a atender tokens más informativos automáticamente.

### 2. Multi-Scale Targets

Predecir múltiples horizontes temporales:

```python
# Target corto (20% siguiente)
target_short = tokens[context_end:context_end+short_span]

# Target medio (40% siguiente)  
target_medium = tokens[context_end:context_end+medium_span]

# Pérdida combinada
loss = 0.5 * vicreg(z_pred_short, z_target_short) + \
       0.5 * vicreg(z_pred_medium, z_target_medium)
```

**Ventaja:** Aprende representaciones multi-escala (útil para razonamiento jerárquico).

### 3. Negative Mining en SIGReg

Aumentar contraste con ejemplos negativos hard:

```python
# Seleccionar ejemplos con alta self-information
hard_negatives = target_ids[self_info > percentile_95]

# Aumentar peso en estos casos
weights[hard_negatives_mask] *= 2.0
```

---

## 📊 Protocolo de Evaluación Rigurosa

### Métricas Cuantitativas

1. **Representation Quality Index (RQI)**
   ```
   RQI = (Variance_Score × Decorrelation_Score × Alignment_Score)^(1/3)
   
   Donde:
   - Variance_Score = min(variance/γ, 1.0)
   - Decorrelation_Score = 1 - (off_diag_power / d²)
   - Alignment_Score = 1 - (invariance_loss / baseline)
   ```

2. **Latent Space Isotropy**
   ```
   Isotropy = std(eigenvalues) / mean(eigenvalues)
   
   Isotropía perfecta = 0 (todas las dimensiones tienen igual varianza)
   ```

3. **Semantic Preservation Test**
   - Crear pares (contexto, target) con labels semánticos conocidos
   - Medir accuracy de k-NN en espacio latente
   - Target: >80% para considerarse "semánticamente preservador"

### Evaluación Cualitativa

1. **Interpolación Latente**
   ```python
   z1 = encode("El gato duerme")
   z2 = encode("El perro corre")
   
   for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
       z_interp = (1-alpha)*z1 + alpha*z2
       # Visualizar o decodificar z_interp
   ```

2. **Analogías Semánticas**
   ```python
   # Rey - Hombre + Mujer ≈ Reina
   z_analogy = z_rey - z_hombre + z_mujer
   nearest = find_nearest_in_corpus(z_analogy)
   ```

---

## 🎓 Transferencia a Downstream Tasks

### Fine-Tuning Strategy

El modelo JEPA entrenado sirve como **feature extractor universal**:

```python
# Congelar Student encoder
for param in model.student.parameters():
    param.requires_grad = False

# Añadir cabeza de clasificación
classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, num_classes)
)

# Fine-tune solo la cabeza
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
```

### Tareas Recomendadas

1. **Clasificación de Texto**
   - Sentiment Analysis (SST-2)
   - Topic Classification (AG News)

2. **Question Answering**
   - SQuAD
   - Natural Questions

3. **Semantic Similarity**
   - STS Benchmark
   - SICK dataset

**Expectativa de rendimiento:** Un JEPA bien entrenado debería alcanzar 85-90% de la performance de BERT en estas tareas con 10x menos parámetros.

---

## 🔮 Direcciones Futuras

### 1. Multi-Modal JEPA (Vision + Language)

Extender a pares (imagen, caption):

```python
# Image Encoder: Vision Transformer
image_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224")

# Text Encoder: Qwen (ya tenemos)
text_encoder = model.student

# Predictor cross-modal
cross_predictor = LatentPredictor(input_dim=768+1024, ...)

# Loss: Predecir embedding de texto desde imagen
z_pred = cross_predictor(image_embedding)
vicreg_loss(z_pred, text_embedding)
```

### 2. Hierarchical JEPA (H-JEPA)

Predecir en múltiples niveles de abstracción:

```
Level 1: Token embeddings
Level 2: Sentence embeddings  
Level 3: Paragraph embeddings

Loss = Σ_i w_i · VICReg(z_pred^i, z_target^i)
```

### 3. Online Learning con Continual JEPA

Actualizar el modelo con nuevos datos sin catastrofic forgetting:

```python
# Experience Replay Buffer
replay_buffer = []

for new_batch in stream:
    # Mezclar con ejemplos antiguos
    mixed_batch = sample(replay_buffer, k=0.5*batch_size) + new_batch
    
    # Entrenar
    loss = vicreg(mixed_batch)
    
    # Actualizar buffer
    replay_buffer.append(new_batch)
```

---

## 📚 Referencias y Fundamentos Teóricos

### Papers Fundamentales

1. **VICReg (Bardes et al., 2022)**
   - "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
   - Aportación: Regularización no-contrastiva sin proyección negativa

2. **I-JEPA (LeCun et al., 2023)**
   - "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
   - Aportación: Predicción en espacio latente para visión

3. **EMA Teacher (Caron et al., 2021 - DINO)**
   - "Emerging Properties in Self-Supervised Vision Transformers"
   - Aportación: Estabilización mediante momentum encoder

### Conexión con Teoría Matemática

**Control Theory Perspective:**
El sistema Student-Teacher-Predictor es un **control adaptativo** donde:
- Student = Sistema controlado
- Teacher = Referencia (setpoint)
- Predictor = Controlador
- VICReg = Función de costo

**Information Theory Perspective:**
La arquitectura maximiza **mutual information** I(X;Y) mientras minimiza **redundancy** entre dimensiones:

```
max I(X;Y) - β·R(Z)
donde R(Z) es la redundancia medida por correlación
```

---

## 🛡️ Conclusiones y Best Practices

### Checklist de Implementación Robusta

- [ ] BatchNorm en Predictor (NO LayerNorm)
- [ ] Stop-gradient explícito en Teacher
- [ ] Epsilon numérico en cálculos de std/var
- [ ] Gradient clipping habilitado
- [ ] EMA schedule coseno implementado
- [ ] Logging de todas las sub-componentes de loss
- [ ] Checkpoint saving cada N steps
- [ ] Evaluación periódica con T-SNE
- [ ] Monitoring de collapse metrics

### Hiperparámetros Críticos (No Modificar sin Evidencia)

```python
CRITICAL_HYPERPARAMS = {
    'lambda_inv': 25.0,     # Basado en VICReg paper
    'mu_var': 25.0,         # Balance crítico con invariance
    'nu_cov': 1.0,          # Suficiente para decorrelación
    'gamma': 1.0,           # Estándar en literatura SSL
    'ema_base': 0.996,      # Validado en DINO, MoCo
    'batch_size_min': 32,   # Mínimo para covarianza estable
}
```

### Debugging Workflow

```
1. Verificar shapes de tensores
2. Plotear histogramas de activaciones
3. Verificar que gradientes fluyen (torch.autograd.grad)
4. Comparar con implementación de referencia (VICReg oficial)
5. Reducir modelo a versión toy (d=64) para diagnóstico rápido
```

---

**Este documento complementa el notebook de implementación con fundamentos matemáticos rigorosos y consideraciones de ingeniería de producción.**

**Arquitectura diseñada para:** Investigadores, ingenieros de ML, y equipos de producto que buscan implementar modelos de mundo en producción con garantías de estabilidad y convergencia.
