"""
XAI with ANM - Demonstration API & Static Web Server.

Serves the frontend demonstration interface from web/ and provides minimal API endpoints:
- GET  /api/info: Returns metadata about the validated UCI Wine dataset and model.pkl.
- POST /api/analyze: Executes the existing AML ANM+HSIC causal discovery pipeline
                     (from generate_benchmark.py) and CAI attribution (from cai.py)
                     on wine.csv and model.pkl without duplicating or altering any algorithms.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple
import urllib.parse

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import existing untouched AML pipeline functions and classes
from generate_benchmark import run_discovery_pipeline
from cai import CausalANM, CausalFeatureAttributor

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
WINE_CSV_PATH = BASE_DIR / "wine.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

DEFAULT_PORT = 8000


def to_json_serializable(val: Any) -> Any:
    """Recursively converts NumPy scalars and arrays to JSON-compatible native types."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, np.ndarray):
        return [to_json_serializable(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [to_json_serializable(x) for x in val]
    if isinstance(val, dict):
        return {k: to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, set):
        return sorted([to_json_serializable(x) for x in val])
    return val


def get_dataset_info() -> Dict[str, Any]:
    """Inspects the local wine.csv and model.pkl to return actual metadata."""
    df = pd.read_csv(WINE_CSV_PATH)
    features = list(df.columns)
    n_samples, n_features = df.shape
    n_possible_pairs = n_features * (n_features - 1) // 2

    model_type = "LogisticRegression"
    n_classes = 3
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            model_type = type(model).__name__
            if hasattr(model, "classes_"):
                n_classes = len(model.classes_)
        except Exception as e:
            print(f"[Warning] Failed inspecting model.pkl: {e}", file=sys.stderr)

    return {
        "dataset_name": "Wine Dataset (UCI)",
        "csv_file": "wine.csv",
        "n_samples": n_samples,
        "n_features": n_features,
        "feature_names": features,
        "n_possible_pairs": n_possible_pairs,
        "model_file": "model.pkl",
        "model_type": model_type,
        "n_classes": n_classes,
        "explained_target": "P(class 0)",
        "default_percentile": 75.0,
        "pipeline_seed": 42,
    }


def execute_pipeline() -> Dict[str, Any]:
    """
    Executes the validated Wine pipeline by directly calling the existing implementations.
    Parity with ANM_DAG_Discovery.ipynb and the published results:
    - seed = 42
    - percentile = 75.0
    - subsample_size = 500
    - bw_subsample_size = 300
    - K_cai = 1000
    """
    if not WINE_CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {WINE_CSV_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    df = pd.read_csv(WINE_CSV_PATH)
    feature_names = list(df.columns)
    p = len(feature_names)

    # 1. Run causal discovery pipeline using the existing untouched implementation
    discovery_res = run_discovery_pipeline(
        df=df,
        dataset_seed=None,
        pipeline_seed=42,
        n_splits=5,
        percentile=75.0,
        subsample_size=500,
        bw_subsample_size=300,
        ground_truth_edges=None,
    )

    # 2. Build DAG and determine node topological order and structural roles
    G = nx.DiGraph()
    for idx, name in enumerate(feature_names):
        G.add_node(idx, name=name)

    edge_conf_map = {(u, v): conf for u, v, conf in discovery_res.edge_candidates}
    retained_edges_tuples: Set[Tuple[int, int]] = set()

    for u in range(p):
        for v in range(p):
            if discovery_res.adj_final[u, v]:
                G.add_edge(u, v, weight=edge_conf_map.get((u, v), 0.0))
                retained_edges_tuples.add((u, v))

    is_dag = nx.is_directed_acyclic_graph(G)
    topo_indices = list(nx.topological_sort(G)) if is_dag else list(range(p))
    topo_names = [feature_names[i] for i in topo_indices]

    # Node roles: Root (in_degree == 0, out_degree > 0), Sink (in_degree > 0, out_degree == 0),
    # Mediator (in_degree > 0, out_degree > 0), Isolated (in_degree == 0, out_degree == 0)
    dag_nodes = []
    for idx in range(p):
        name = feature_names[idx]
        in_deg = G.in_degree(idx)
        out_deg = G.out_degree(idx)
        if in_deg == 0 and out_deg == 0:
            role = "isolated"
        elif in_deg == 0:
            role = "root"
        elif out_deg == 0:
            role = "sink"
        else:
            role = "mediator"

        dag_nodes.append({
            "id": idx,
            "name": name,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "role": role,
            "topo_rank": topo_indices.index(idx) if idx in topo_indices else -1,
        })

    # Prepare detailed edge records for graph and pairwise inspection
    dag_edges = []
    pair_metric_lookup = {}
    for rec in discovery_res.pairwise_records:
        u = rec["Cause_Idx"]
        v = rec["Effect_Idx"]
        pair_metric_lookup[(u, v)] = rec

    for u, v in retained_edges_tuples:
        rec = pair_metric_lookup.get((u, v), {})
        dag_edges.append({
            "source_idx": u,
            "target_idx": v,
            "source": feature_names[u],
            "target": feature_names[v],
            "confidence": float(rec.get("Confidence_Score", edge_conf_map.get((u, v), 0.0))),
            "forward_hsic": float(rec.get("Forward_HSIC", 0.0)),
            "reverse_hsic": float(rec.get("Reverse_HSIC", 0.0)),
            "direction": f"{feature_names[u]} → {feature_names[v]}",
        })

    # Sort DAG edges by confidence descending
    dag_edges.sort(key=lambda item: item["confidence"], reverse=True)

    # Pairwise records (all 78 evaluated pairs) with full status annotation
    all_pairs = []
    for rec in discovery_res.pairwise_records:
        u = rec["Cause_Idx"]
        v = rec["Effect_Idx"]
        conf = float(rec["Confidence_Score"])
        direction_str = f"{feature_names[u]} → {feature_names[v]}"

        is_retained = (u, v) in retained_edges_tuples
        is_above_threshold = conf >= discovery_res.threshold

        if is_retained:
            status = "Retained in DAG"
        elif is_above_threshold:
            status = "Pruned (Cycle Elimination)"
        else:
            status = "Filtered by Threshold"

        all_pairs.append({
            "pair_index": int(rec["Pair_Index"]),
            "cause": feature_names[u],
            "effect": feature_names[v],
            "cause_idx": u,
            "effect_idx": v,
            "forward_hsic": float(rec["Forward_HSIC"]),
            "reverse_hsic": float(rec["Reverse_HSIC"]),
            "direction": direction_str,
            "confidence": conf,
            "status": status,
            "is_retained": is_retained,
        })

    all_pairs.sort(key=lambda x: x["confidence"], reverse=True)

    # 3. Fit Causal ANM and evaluate CAI using the existing untouched CAI module
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_names].values)

    causal_anm = CausalANM(
        adj=discovery_res.adj_final,
        order=topo_indices,
        random_state=42,
    )
    causal_anm.fit(scaled_data)

    model = joblib.load(MODEL_PATH)

    def prediction_function(samples: np.ndarray) -> np.ndarray:
        return model.predict_proba(np.asarray(samples, dtype=float))[:, 0]

    obs_idx = 0
    x_obs = scaled_data[obs_idx].copy()

    attributor = CausalFeatureAttributor(
        anm=causal_anm,
        prediction_function=prediction_function,
        K_cai=1000,
    )
    cai_res = attributor.attribute(
        x_obs=x_obs,
        feature_names=feature_names,
        verbose=False,
    )

    feature_cai_records = []
    for j in range(p):
        name = feature_names[j]
        cai_val = float(cai_res.cai[j])
        cai_norm_val = float(cai_res.cai_norm[j])
        se_val = float(cai_res.standard_errors[j])
        ci_low = float(cai_res.confidence_intervals[j, 0])
        ci_high = float(cai_res.confidence_intervals[j, 1])
        # Significant if CI does not straddle zero
        is_sig = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)

        feature_cai_records.append({
            "feature": name,
            "feature_idx": j,
            "cai": cai_val,
            "cai_norm": cai_norm_val,
            "standard_error": se_val,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "is_significant": is_sig,
            "role": dag_nodes[j]["role"],
        })

    # Sort CAI descending by raw CAI
    feature_cai_records.sort(key=lambda item: item["cai"], reverse=True)

    return {
        "dataset_info": {
            "name": "Wine Dataset (UCI)",
            "samples": len(df),
            "features": p,
            "feature_names": feature_names,
        },
        "overview": {
            "samples": len(df),
            "features": p,
            "pairs_evaluated": len(discovery_res.pairwise_records),
            "candidate_edges": len(dag_edges),
            "threshold": float(discovery_res.threshold),
            "percentile": 75.0,
            "is_dag": bool(is_dag),
            "dag_status": "Acyclic (Valid DAG)" if is_dag else "Contains Cycles",
            "cycles_removed": int(discovery_res.cycles_removed),
        },
        "topological_order": topo_names,
        "dag_nodes": dag_nodes,
        "dag_edges": dag_edges,
        "retained_edges": [f"{e['source']} → {e['target']}" for e in dag_edges],
        "pairwise_records": all_pairs,
        "cai": {
            "baseline": float(cai_res.baseline),
            "observation_index": obs_idx,
            "explained_target": "P(class 0)",
            "monte_carlo_samples": 1000,
            "records": feature_cai_records,
        },
    }


class AMLHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web/ and JSON endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/info":
            try:
                info = get_dataset_info()
                self._send_json(info)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # Serve static files from web/
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            try:
                results = execute_pipeline()
                clean_results = to_json_serializable(results)
                self._send_json(clean_results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": f"Unknown endpoint: {parsed.path}"}, status=404)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[AML Server] {self.address_string()} - {format % args}")


def run_server(port: int = DEFAULT_PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, AMLHandler)
    print("=" * 65)
    print("   XAI with ANM: Research Demonstration Server")
    print(f"   Serving UI at:   http://localhost:{port}/")
    print(f"   API Endpoints:   GET  http://localhost:{port}/api/info")
    print(f"                    POST http://localhost:{port}/api/analyze")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[AML Server] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    run_server(port)
