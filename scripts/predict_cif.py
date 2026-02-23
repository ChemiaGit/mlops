#!/usr/bin/env python3
"""
Predict a material property from a CIF file using the cgcnn-inference endpoint.

Usage:
    python scripts/predict_cif.py --cif temp/BaAg(PO3)3.cif --property specific-heat
    python scripts/predict_cif.py --cif temp/BaAg(PO3)3.cif --property t-debye
    python scripts/predict_cif.py --cif temp/BaAg(PO3)3.cif --property dielectric

Available properties:
    specific-heat   Specific heat capacity
    t-debye         Debye temperature
    dielectric      Dielectric constant
    seebeck-n       Seebeck coefficient (n-type)
    seebeck-p       Seebeck coefficient (p-type)
    zt-n            Thermoelectric figure of merit ZT (n-type)
    zt-p            Thermoelectric figure of merit ZT (p-type)
"""

import argparse
import json
import os
import tempfile

PROPERTIES = [
    "specific-heat",
    "t-debye",
    "dielectric",
    "seebeck-n",
    "seebeck-p",
    "zt-n",
    "zt-p",
]


def main():
    parser = argparse.ArgumentParser(description="Predict material property from CIF")
    parser.add_argument("--cif", required=True, help="Path to .cif file")
    parser.add_argument(
        "--property", required=True, choices=PROPERTIES,
        metavar="PROPERTY",
        help=f"Property to predict. Choices: {', '.join(PROPERTIES)}",
    )
    args = parser.parse_args()

    cif_path = os.path.abspath(args.cif)
    if not os.path.isfile(cif_path):
        print(f"ERROR: CIF file not found: {cif_path}")
        raise SystemExit(1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    ml = MLClient(
        DefaultAzureCredential(),
        subscription_id="5350c60c-7da3-4391-9272-2edf3e07b3c9",
        resource_group_name="baghsev-rg",
        workspace_name="BandStructureLearning",
    )

    cif_text = open(cif_path).read()
    material_id = os.path.splitext(os.path.basename(cif_path))[0]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"id": material_id, "cif": cif_text}, f)
        tmp = f.name

    try:
        result = ml.online_endpoints.invoke(
            endpoint_name="cgcnn-inference",
            deployment_name=args.property,
            request_file=tmp,
        )
    finally:
        os.unlink(tmp)

    data = json.loads(json.loads(result) if isinstance(result, str) and result.startswith('"') else result)
    print(f"Material:  {data['id']}")
    print(f"Property:  {data['property']}")
    print(f"Prediction: {data['prediction']:.6f}")


if __name__ == "__main__":
    main()
