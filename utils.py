from collections import defaultdict
from html import escape
from math import cos, pi, sin, sqrt
from pathlib import Path
from textwrap import fill
from time import perf_counter
from types import SimpleNamespace
import json
import re

import pandas as pd
from tqdm.auto import tqdm



def _positions(graph, aspect=1.6):
    """Lay out each connected component, then pack their bounding boxes into rows."""
    import networkx as nx

    boxes = []
    for members in nx.connected_components(graph):
        part = graph.subgraph(members)
        if len(part) == 1:
            points = {next(iter(part)): (0, 0)}
        elif len(part) <= 120:
            points = nx.spring_layout(part, seed=42, iterations=35, scale=46 * sqrt(len(part)))
        else:
            # A breadth-first spiral keeps large components compact without a force simulation.
            root = max(sorted(part), key=part.degree.__getitem__)
            points = {}
            for i, node in enumerate(nx.bfs_tree(part, root)):
                angle = i * pi * (3 - sqrt(5))
                points[node] = (40 * sqrt(i) * cos(angle), 40 * sqrt(i) * sin(angle))

        xs, ys = zip(*points.values())
        width, height = max(xs) - min(xs) + 100, max(ys) - min(ys) + 70
        boxes.append((width, height, min(xs), min(ys), points))

    target_width = sqrt(aspect * sum(w * h for w, h, *_ in boxes))
    result = {}
    x = y = row_height = 0
    for w, h, left, top, points in sorted(boxes, key=lambda box: -box[1]):
        if x and x + w > target_width:
            x, y, row_height = 0, y + row_height, 0
        for node, (px, py) in points.items():
            result[node] = (round(x + px - left), round(y + py - top))
        x += w
        row_height = max(row_height, h)
    return result


def _write_view(path, nodes, triples, *, detail, question):
    import networkx as nx
    from pyvis.network import Network

    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    predicates = defaultdict(list)
    for source, relation, target in sorted(triples):
        graph.add_edge(source, target)
        predicates[source, target].append(relation)

    net = Network(height="100vh", width="100%", directed=True, cdn_resources="in_line")
    positions = _positions(graph, aspect=5 if detail else 1.6)
    for node_id, (x, y) in positions.items():
        node = nodes[node_id]
        is_chunk = hasattr(node, "text")
        name = "Source chunk" if is_chunk else node_id
        tooltip = f"{node_id}\n{node.text[:350]}" if is_chunk else name
        if detail:
            net.add_node(node_id, label=fill(name, width=18), title=escape(tooltip),
                         x=2 * x, y=2 * y, shape="box", margin=9,
                         color={"background": "#f1f8f8", "border": "#257a86"},
                         font={"size": 24, "color": "#183c42"})
        else:
            net.add_node(node_id, label=name if len(name) <= 40 else name[:39] + "…",
                         title=escape(tooltip), x=x, y=y, shape="square" if is_chunk else "dot",
                         color="#bbc3ce" if is_chunk else "#257a86", size=6 if is_chunk else 10)

    for (source, target), names in predicates.items():
        label = " / ".join(names)
        # Opposite directions curve away from one another; parallel predicates share a tooltip.
        curve = {"enabled": True, "type": "curvedCW", "roundness": 0.18} if (target, source) in predicates else False
        net.add_edge(source, target, arrows="to", smooth=curve, title=escape(label),
                     label=fill(label, width=14) if detail else "", color="#8795a2", width=1)

    net.set_options(json.dumps({
        "physics": {"enabled": False}, "layout": {"improvedLayout": False},
        "interaction": {"hover": True, "hideEdgesOnDrag": not detail},
        "edges": {"font": {"size": 16 if detail else 11, "align": "middle"}, "arrows": {"to": {"scaleFactor": 0.45}}},
    }))
    html = net.generate_html(notebook=False)
    # PyVis embeds vis-network, but its stock template also links optional Bootstrap assets.
    html = re.sub(r'<link\b[^<>]*\bhref="https?://[^"]+"[^>]*>', "", html, flags=re.I)
    html = re.sub(r'<script\b[^<>]*\bsrc="https?://[^"]+"[^>]*>\s*</script>', "", html, flags=re.I)
    title = "Retrieved relation hints" if detail else "Full knowledge graph"
    scope = "Relations included in the final answer context; retrieved dialogue is not shown." if detail else "All stored nodes and relations; gray squares are source chunks."
    banner = f'<div id="graph-info"><strong>{title}</strong> · {len(nodes):,} nodes · {len(triples):,} relations<br>{escape(question)}<br><small>{scope}</small></div>'
    style = "<style>html,body{margin:0;overflow:hidden}#graph-info{font:14px/20px Arial,sans-serif;padding:10px 16px}#mynetwork{position:fixed;inset:80px 0 0;height:auto;border:0}</style>"
    html = html.replace("</head>", style + "</head>").replace("<body>", "<body>" + banner)
    path.write_text(html, encoding="utf-8")


def export_memory_graphs(response, output_dir="."):
    """Write graph_full.html and graph_retrieved.html from one saved §6 response."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    store = response["graph_store"]
    nodes = {node.id: node for node in store.get()}
    triples = set(store.graph.triplets)
    by_line = defaultdict(list)
    for triple in triples:
        by_line[" -> ".join(triple)].append(triple)

    selected = set()
    preamble = "Here are some facts extracted from the provided text:"
    for evidence in response["retrieved"]:
        text = evidence.strip()
        if text.startswith("Conversation excerpts:"):
            _, marker, relations = text.rpartition("\n\nGraph relations:\n")
            if not marker:
                continue
            lines = relations.splitlines()
        elif text.startswith("Graph relations:\n"):
            lines = text[len("Graph relations:\n"):].splitlines()
        elif text.startswith(preamble):
            lines = text[len(preamble):].lstrip().partition("\n\n")[0].splitlines()
        else:
            lines = [text]  # A graph-only hit has one relation and no attached dialogue.
        for line in lines:
            matches = by_line.get(line.strip(), [])
            if len(matches) != 1:
                raise ValueError(f"Cannot match retrieved relation uniquely in this store: {line[:130]!r}")
            selected.add(matches[0])

    full = directory / "graph_full.html"
    retrieved = directory / "graph_retrieved.html"
    selected_nodes = {node for source, _, target in selected for node in (source, target)}
    _write_view(full, nodes, triples, detail=False, question=response["question"])
    _write_view(retrieved, {node: nodes[node] for node in selected_nodes}, selected, detail=True, question=response["question"])
    return full, retrieved
