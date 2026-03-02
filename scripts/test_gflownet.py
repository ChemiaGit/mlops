#!/usr/bin/env python3
"""
Test the gflownet-inference endpoint and display results.

Usage:
    # Generate 5 structures (default)
    python scripts/test_gflownet.py

    # Generate with custom parameters
    python scripts/test_gflownet.py --n-samples 10 --temperature 0.5

    # Score a specific Si/Al config
    python scripts/test_gflownet.py --score 0 1 0 0 1 0 1 0 1 0 0 0

    # Use a different deployment (e.g. for blue/green)
    python scripts/test_gflownet.py --deployment gfn-v2
"""

import argparse
import json
import os
import sys
import tempfile

ENDPOINT_NAME   = "gflownet-inference"
WORKSPACE       = "mlw-gflownet-dev-001"
RESOURCE_GROUP  = "gflownet-vlad"
SUBSCRIPTION_ID = "5350c60c-7da3-4391-9272-2edf3e07b3c9"


def build_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )


def invoke(ml_client, payload: dict, deployment: str | None) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp = f.name
    try:
        raw = ml_client.online_endpoints.invoke(
            endpoint_name=ENDPOINT_NAME,
            deployment_name=deployment,
            request_file=tmp,
        )
    finally:
        os.unlink(tmp)

    # SDK may double-encode the result as a JSON string
    data = json.loads(raw)
    if isinstance(data, str):
        data = json.loads(data)
    return data


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_generate(data: dict):
    samples = data.get("samples", [])
    print(f"\nGenerated {len(samples)} structure(s)\n")
    print(f"{'#':<4} {'Si/Al':>6} {'nAl':>4} {'traj':>5} {'reward':>10} {'energy':>10}  config")
    print("-" * 90)
    for i, s in enumerate(samples):
        if "error" in s:
            print(f"{i+1:<4} ERROR: {s['error']}")
            continue
        reward   = f"{s['reward']:.4f}"   if "reward"       in s else "—"
        energy   = f"{s['proxy_energy']:.4f}" if "proxy_energy" in s else "—"
        si_al    = f"{s.get('si_al_ratio', 0):.2f}"
        n_al     = s.get("n_substitutions", "?")
        traj_len = s.get("trajectory_length", "?")
        config   = "".join(str(v) for v in s.get("config", []))
        print(f"{i+1:<4} {si_al:>6} {n_al:>4} {traj_len:>5} {reward:>10} {energy:>10}  {config}")

    if len(samples) > 1:
        rewards = [s["reward"] for s in samples if "reward" in s]
        if rewards:
            print(f"\n  Best reward : {max(rewards):.6f}")
            print(f"  Mean reward : {sum(rewards)/len(rewards):.6f}")
            best = max(samples, key=lambda s: s.get("reward", -1))
            print(f"  Best config : {''.join(str(v) for v in best.get('config', []))}")
            print(f"  Best Si/Al  : {best.get('si_al_ratio', '?'):.2f}")


def print_score(data: dict):
    config = "".join(str(v) for v in data.get("config", []))
    print(f"\nScored configuration")
    print(f"  Config      : {config}")
    print(f"  Si/Al ratio : {data.get('si_al_ratio', '?'):.4f}")
    print(f"  Al count    : {data.get('n_substitutions', '?')}")
    if "reward" in data:
        print(f"  Reward      : {data['reward']:.6f}")
    if "proxy_energy" in data:
        print(f"  Proxy energy: {data['proxy_energy']:.6f}")
    if "policy_entropy" in data:
        print(f"  Policy H    : {data['policy_entropy']:.4f}  (higher = more uncertain)")
    if "proxy_error" in data:
        print(f"  Proxy error : {data['proxy_error']}")


# ---------------------------------------------------------------------------
# CIF export
# ---------------------------------------------------------------------------

def save_cifs(samples: list, output_dir: str):
    """Write CIF strings returned by the endpoint directly to disk."""
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, s in enumerate(samples):
        if "error" in s or "cif" not in s:
            continue
        si_al   = s.get("si_al_ratio", 0.0)
        reward  = s.get("reward", 0.0)
        cfg_str = "".join(str(v) for v in s.get("config", []))
        fname   = f"structure_{i+1:03d}_SiAl{si_al:.2f}_R{reward:.4f}_{cfg_str}.cif"
        fpath   = os.path.join(output_dir, fname)
        with open(fpath, "w") as f:
            f.write(s["cif"])
        print(f"  [{i+1}] {fname}")
        saved += 1

    print(f"\nSaved {saved} CIF file(s) to {os.path.abspath(output_dir)}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the gflownet-inference endpoint"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--score", nargs="+", type=int, metavar="BIT",
        help="Score a specific config (binary Si/Al array). "
             "E.g. --score 0 1 0 0 1 0 1 0"
    )
    mode_group.add_argument(
        "--n-samples", type=int, default=5,
        help="Number of structures to generate (default: 5)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for generation (default: 1.0). "
             "Lower = more greedy, higher = more exploratory."
    )
    parser.add_argument(
        "--deployment", default=None,
        help="Deployment name (default: active deployment). E.g. gfn-v1"
    )
    parser.add_argument(
        "--json", action="store_true", dest="raw_json",
        help="Print raw JSON response instead of formatted output"
    )
    parser.add_argument(
        "--save-cif", metavar="DIR", default=None,
        help="Save generated structures as CIF files to DIR (generate mode only)"
    )
    args = parser.parse_args()

    try:
        from azure.ai.ml import MLClient  # noqa: F401
    except ImportError:
        print("ERROR: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    ml_client = build_client()

    if args.score is not None:
        payload = {"mode": "score", "config": args.score}
        print(f"Scoring config: {''.join(str(b) for b in args.score)}")
    else:
        payload = {
            "mode": "generate",
            "n_samples": args.n_samples,
            "temperature": args.temperature,
        }
        print(f"Generating {args.n_samples} structure(s) at temperature={args.temperature} ...")

    data = invoke(ml_client, payload, args.deployment)

    if args.raw_json:
        print(json.dumps(data, indent=2))
        return

    if data.get("mode") == "generate":
        print_generate(data)
        if args.save_cif:
            save_cifs(data.get("samples", []), args.save_cif)
    elif data.get("mode") == "score":
        print_score(data)
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
