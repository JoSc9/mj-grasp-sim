#!/usr/bin/env python3
"""Visualize a labeled point cloud with Plotly and serve it on localhost.

Usage:
  python scripts/visualize_pcd_plotly.py /path/to/scene_pcd.npz [--port 8000]

This writes a temporary HTML file (next to the .npz) and starts a simple
HTTP server on the given port, then opens the HTML in the default browser.
"""
import argparse
import os
import time
import webbrowser
# no HTTP server needed; Plotly HTML can be opened directly

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


def hex_to_rgb_tuple(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def build_color_array(labels):
    n = labels.shape[0]
    if labels is None:
        return ["rgb(128,128,128)"] * n
        # colors may be floats in [0,1] or 0-255

    uniq = np.unique(labels)
    palette =["red", "green", "blue", "goldenrod", "magenta"]
    # map each unique label to a palette color (cycle if needed)
    color_labels = np.empty_like(labels, dtype=object) 
    for i, u in enumerate(uniq):
        color_labels[labels == u] = palette[i % len(palette)]
    return color_labels

def main():
    p = argparse.ArgumentParser(description="Serve a Plotly 3D view of a labeled point cloud")
    p.add_argument("npz", help="path to scene_pcd.npz")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    pts = data["points"].reshape(-1, 3)
    cols = data.get("colors", None)
    labels = data.get("labels", None)

    color_array = build_color_array(labels)
    idx=5000
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:idx, 0],
                y=pts[:idx, 1],
                z=pts[:idx, 2],
                mode="markers",
                marker=dict(size=2, color=color_array[:idx]),
            )
        ]
    )
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0))
    fig.show(renderer="browser")


if __name__ == "__main__":
    main()
