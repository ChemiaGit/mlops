# chemia/mlops — Model Registry & Inference Pipeline

Azure ML workflow for Chemia ML models (CGCNNs + GFlowNet).

```
mlops/
├── register_notebook_models.py     # Register CGCNN .pt files from models_storage/
├── register_existing_models.py     # Register GFlowNet checkpoints from past AML jobs
│
├── inference/
│   ├── cgcnn/
│   │   ├── score.py                # CGCNN scoring script (CIF → property value)
│   │   ├── cgcnn_model.py          # Standalone PyTorch model (DeepChem-compatible)
│   │   ├── Dockerfile              # Custom environment: PyTorch 1.13 + pymatgen
│   │   └── conda.yml               # Conda spec (reference)
│   └── gflownet/
│       ├── score.py                # GFlowNet scoring script (generate / score configs)
│       └── gflownet_src/           # Bundled mat-gen source (copy from mat-gen/src/gflownet)
│
├── endpoints/
│   ├── create_endpoint.py          # Create/manage managed online endpoints
│   ├── create_cgcnn_environment.py # Build and register cgcnn-inference-env Docker image
│   ├── deploy_all_cgcnn.py         # Deploy all 7 CGCNN models in one command
│   └── deploy_model.py             # Deploy / update / delete individual deployments
│
├── scripts/
│   └── predict_cif.py              # CLI: predict any property from a .cif file
│
├── sample_inputs/
│   ├── cgcnn_sample.json           # Single crystal structure (NaCl CIF)
│   ├── cgcnn_batch_sample.json     # Multiple structures
│   ├── gflownet_generate.json      # Generate N zeolite configs
│   └── gflownet_score.json         # Score a given Si/Al config
│
└── requirements.txt
```

---

# CGCNN — Property Prediction

Models predict material properties from crystal structure (CIF input).
All models share the same architecture and scoring script — only weights differ.

## Registered models

| Deployment name | Registry name                          | Version |
|-----------------|----------------------------------------|---------|
| specific-heat   | CGCNN_specific_heat                    | 1       |
| t-debye         | CGCNN_t_debye                          | 1       |
| dielectric      | CGCNN_dielectric_constant              | 1       |
| seebeck-n       | CGCNN_thermoelectric_seebeck_n         | 1 (best val loss: 11.37) |
| seebeck-p       | CGCNN_thermoelectric_seebeck_p         | 1 (best val loss: 9.52)  |
| zt-n            | CGCNN_thermoelectric_electronic_zt_n   | 1       |
| zt-p            | CGCNN_thermoelectric_electronic_zt_p   | 1       |

## 1. Register CGCNN models

```bash
# Register a specific folder (name = folder minus timestamp)
python register_notebook_models.py \
    --folder CGCNN_specific_heat_1757119144 \
    --stage production

# Register all CGCNN folders at once
python register_notebook_models.py --all
```

## 2. Build the inference environment (once)

The CGCNN scoring script requires `pymatgen`, which is not in the curated AML
environment. Build a custom Docker image on top of the PyTorch base:

```bash
python endpoints/create_cgcnn_environment.py \
    --workspace BandStructureLearning \
    --resource-group baghsev-rg \
    --version 2
```

Monitor the build in AML Studio → Environments → cgcnn-inference-env.
Takes ~5–10 minutes. Only needs to be re-run if the Dockerfile changes.

## 3. Deploy all models (Option A — one endpoint, named deployments)

```bash
# Create endpoint (once)
python endpoints/create_endpoint.py \
    --name cgcnn-inference \
    --workspace BandStructureLearning \
    --resource-group baghsev-rg

# Deploy all 7 models
python endpoints/deploy_all_cgcnn.py \
    --workspace BandStructureLearning \
    --resource-group baghsev-rg \
    --environment cgcnn-inference-env:2
```

This creates one deployment per property. Default traffic routes to `specific-heat`.
All other properties are reached via the `azureml-model-deployment` request header.

```
cgcnn-inference endpoint
  ├── specific-heat   ← 100% default traffic
  ├── t-debye
  ├── dielectric
  ├── seebeck-n
  ├── seebeck-p
  ├── zt-n
  └── zt-p
```

## 4. Predict from a CIF file

```bash
python scripts/predict_cif.py --cif temp/BaAg\(PO3\)3.cif --property specific-heat
python scripts/predict_cif.py --cif temp/BaAg\(PO3\)3.cif --property t-debye
python scripts/predict_cif.py --cif temp/BaAg\(PO3\)3.cif --property seebeck-n
```

Output:
```
Material:   BaAg(PO3)3
Property:   specific_heat
Prediction: 3.598697
```

Available properties: `specific-heat`, `t-debye`, `dielectric`, `seebeck-n`, `seebeck-p`, `zt-n`, `zt-p`

## 5. Switch to a different model version

**In-place update** — replace the running deployment with a new model version:

```bash
python endpoints/deploy_model.py \
    --endpoint cgcnn-inference \
    --model CGCNN_thermoelectric_seebeck_n:3 \
    --deployment-name seebeck-n \
    --scoring-dir ./inference/cgcnn \
    --environment cgcnn-inference-env:2 \
    --env-vars MODEL_PROPERTY=seebeck_n
```

**Blue/green** — deploy alongside, test, then delete old:

```bash
# Deploy new version under a different name
python endpoints/deploy_model.py \
    --endpoint cgcnn-inference \
    --model CGCNN_thermoelectric_seebeck_n:3 \
    --deployment-name seebeck-n-v3 \
    --scoring-dir ./inference/cgcnn \
    --environment cgcnn-inference-env:2 \
    --env-vars MODEL_PROPERTY=seebeck_n

# Test it
python scripts/predict_cif.py --cif temp/structure.cif --property seebeck-n-v3

# Delete old deployment when satisfied
python endpoints/deploy_model.py \
    --endpoint cgcnn-inference \
    --deployment-name seebeck-n \
    --delete
```

**List all deployments and current versions:**

```bash
python endpoints/deploy_model.py --endpoint cgcnn-inference --list
```

## 6. Direct SDK / HTTP calls

**Python SDK:**
```python
import json, tempfile, os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml = MLClient(
    DefaultAzureCredential(),
    subscription_id="5350c60c-7da3-4391-9272-2edf3e07b3c9",
    resource_group_name="baghsev-rg",
    workspace_name="BandStructureLearning",
)

cif_text = open("path/to/structure.cif").read()
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"id": "my_material", "cif": cif_text}, f)
    tmp = f.name

result = ml.online_endpoints.invoke(
    endpoint_name="cgcnn-inference",
    deployment_name="seebeck-n",   # specific-heat | t-debye | dielectric | ...
    request_file=tmp,
)
os.unlink(tmp)
print(result)
# → {"id": "my_material", "prediction": 3.598697, "property": "seebeck_n"}
```

**curl:**
```bash
KEY=$(az ml online-endpoint get-credentials \
    --name cgcnn-inference \
    --workspace-name BandStructureLearning \
    --resource-group baghsev-rg \
    --query primaryKey -o tsv)

curl -X POST https://cgcnn-inference.eastus2.inference.ml.azure.com/score \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -H "azureml-model-deployment: seebeck-n" \
    -d '{"id": "my_material", "cif": "<cif_text>"}'
```

## 7. Stop endpoint to save cost

Endpoints bill by the hour (~$0.53/hr per T4 deployment × 7 = ~$3.70/hr total).
Delete when not in use — model versions stay in the registry:

```bash
az ml online-endpoint delete --name cgcnn-inference \
    --workspace-name BandStructureLearning \
    --resource-group baghsev-rg \
    --yes
```

Recreate with steps 3–4 above.

---

# GFlowNet — Zeolite Structure Generation

Generates or evaluates Faujasite (FAU) Si/Al substitution configurations.
Source model: `mat-gen/src/gflownet/`

## Workspace

GFlowNet lives in a separate workspace:
- **Workspace**: `mlw-gflownet-dev-001`
- **Resource group**: `gflownet-vlad`

## 1. Register GFlowNet checkpoints

```bash
python register_existing_models.py \
    --workspace mlw-gflownet-dev-001 \
    --resource-group gflownet-vlad \
    --model-name gfn-zeolite-model \
    --checkpoint-iterations 300
```

## 2. Bundle model source

The scoring script imports from `mat-gen`. Copy the source into the inference
directory so it ships with the deployment:

```bash
cp -r /home/vladi/chemia/mat-gen/src/gflownet \
    ./inference/gflownet/gflownet_src/
```

## 3. Build inference environment (once)

The scoring script requires `ase` which is not in the curated AML environment:

```bash
python endpoints/create_gflownet_environment.py \
    --workspace mlw-gflownet-dev-001 \
    --resource-group gflownet-vlad
```

Monitor build in AML Studio → Environments → gflownet-inference-env.

## 4. Deploy

Deployment names must be ≥ 3 characters — use `gfn-v1`, `gfn-v2`, etc.

```bash
# Create endpoint (once)
python endpoints/create_endpoint.py \
    --name gflownet-inference \
    --workspace mlw-gflownet-dev-001 \
    --resource-group gflownet-vlad

# Deploy model
python endpoints/deploy_model.py \
    --endpoint gflownet-inference \
    --model gfn-zeolite-model:6 \
    --deployment-name gfn-v1 \
    --scoring-dir ./inference/gflownet \
    --environment gflownet-inference-env:1 \
    --env-vars GFN_HIDDEN_DIM=256 GFN_NUM_LAYERS=3 \
    --workspace mlw-gflownet-dev-001 \
    --resource-group gflownet-vlad
```

## 5. Test the endpoint

```bash
# Generate 5 structures (default)
python scripts/test_gflownet.py

# Generate with custom parameters
python scripts/test_gflownet.py --n-samples 10 --temperature 0.5

# Score a specific Si/Al config (binary array, length = num_t_sites)
python scripts/test_gflownet.py --score 0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 0 0 1

# Raw JSON output
python scripts/test_gflownet.py --n-samples 3 --json
```

Example output:
```
Generated 5 structure(s)

#     Si/Al  nAl  traj     reward     energy  config
──────────────────────────────────────────────────────────────────────────────────────────
1      1.57    7     8     0.0000     1.3808  011101010001010000
2      1.25    8     9     0.0000     2.3730  011001011010001100
...
  Best reward : 0.000006
  Best Si/Al  : 1.57
```

## 6. Call the endpoint directly (curl / SDK)

**Python SDK:**
```python
import json, tempfile, os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml = MLClient(
    DefaultAzureCredential(),
    subscription_id="5350c60c-7da3-4391-9272-2edf3e07b3c9",
    resource_group_name="gflownet-vlad",
    workspace_name="mlw-gflownet-dev-001",
)

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"mode": "generate", "n_samples": 3, "temperature": 1.0}, f)
    tmp = f.name

result = ml.online_endpoints.invoke(endpoint_name="gflownet-inference", request_file=tmp)
os.unlink(tmp)
print(json.loads(result))
```

**curl:**
```bash
KEY=$(az ml online-endpoint get-credentials \
    --name gflownet-inference \
    --workspace-name mlw-gflownet-dev-001 \
    --resource-group gflownet-vlad \
    --query primaryKey -o tsv)

curl -X POST https://gflownet-inference.eastus2.inference.ml.azure.com/score \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d '{"mode": "generate", "n_samples": 3}'
```

Input formats:
```json
// Generate
{"mode": "generate", "n_samples": 5, "temperature": 1.0}

// Score
{"mode": "score", "config": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]}
```
