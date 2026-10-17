#!/usr/bin/env python3
"""Build the research results site from one content file and the report batches.

Inputs (read only):

* ``docs/site/content.json``: tabs (papers, log, questions), claims, sections.
* ``out/research/report/index.json`` and every batch folder it lists
  (numbered PNGs, ``README.md`` with one ``- `NN_x.png`: description;
  sources: ...`` line per figure, ``tables.md`` with ``### heading`` tables).
* ``out/research/queue.txt`` and ``out/research/done.txt`` (run states).
* JSON result files named by ``per_device`` blocks.

Output: ``out/research/site/index.html`` (page content only, for publishing
as an artifact) and ``out/research/site/figs/<batch>/<file>.png`` (copies, or
symlinks with ``--link``).  Prints a JSON map ``{published_path: source_path}``
of every image file (source paths relative to the repository root) and writes
the same map to ``out/research/site/files.json``.

  .venv/bin/python scripts/research/build_site.py

Content model (see ``_schema`` in content.json):

* tab ``state``: ``active`` | ``merged`` | ``archived``.  A merged tab's claims
  and sections render in ``merged_into``; its id stays a deep-link alias.
  Archived tabs render after the active ones with a notice.
* claim: ``id``, ``tab``, ``statement``, ``status`` (verified | in_progress |
  refuted | planned), ``headline`` {value, convention}, ``numbers``, ``what``,
  ``did``, ``result`` {text, figures, tables, blocks}, ``caveats``,
  ``sources``, ``updated``.  Rendered as ``<details>`` with id
  ``claim-<id>``.
* blocks (sections and claim results): text, list, figure, table,
  batch_table, per_device, timeline, nodes, queue, facts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cryoml import strict_score as ss  # noqa: E402  (result reader only)

STATUS_LABEL = {
    "verified": "verified",
    "in_progress": "in progress",
    "refuted": "refuted",
    "planned": "planned",
}


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers

def esc(text) -> str:
    return html.escape(str(text), quote=True)


_INLINE = re.compile(r"`([^`]+)`|\*\*([^*]+)\*\*|\[([^\]]+)\]\(([^)\s]+)\)")


def inline(text) -> str:
    """Escape text; render `code`, **bold** and [label](href)."""
    text = str(text)
    out, pos = [], 0
    for m in _INLINE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        if m.group(1) is not None:
            out.append(f"<code>{esc(m.group(1))}</code>")
        elif m.group(2) is not None:
            out.append(f"<strong>{esc(m.group(2))}</strong>")
        else:
            href = m.group(4)
            ext = href.startswith(("http://", "https://"))
            attrs = ' target="_blank" rel="noopener"' if ext else ""
            out.append(f'<a href="{esc(href)}"{attrs}>{esc(m.group(3))}</a>')
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def src_line(sources, label="Source") -> str:
    sources = as_list(sources)
    if not sources:
        return ""
    paths = " ".join(f"<code>{esc(s)}</code>" for s in sources)
    lab = label if len(sources) == 1 else label + "s"
    return f'<p class="src"><span class="src-k">{lab}</span> {paths}</p>'


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower()


# ---------------------------------------------------------------- batches

_FIG_LINE = re.compile(r"^- `([^`]+\.png)`: (.*)$")


def parse_readme(path: Path) -> dict:
    figs = {}
    if not path.exists():
        return figs
    for line in path.read_text().splitlines():
        m = _FIG_LINE.match(line.strip())
        if not m:
            continue
        body = m.group(2)
        desc, _, srcs = body.partition("; sources: ")
        figs[m.group(1)] = {
            "description": desc.strip(),
            "sources": re.findall(r"`([^`]+)`", srcs),
        }
    return figs


def parse_tables(path: Path) -> list:
    """Return [{heading, columns, align, rows, notes}] from a tables.md."""
    tables = []
    if not path.exists():
        return tables
    cur = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("### "):
            cur = {"heading": line[4:].strip(), "columns": [], "align": [],
                   "rows": [], "notes": []}
            tables.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not cur["columns"]:
                cur["columns"] = cells
            elif not cur["align"] and all(re.fullmatch(r":?-+:?", c) for c in cells):
                cur["align"] = ["right" if c.endswith(":") else "left" for c in cells]
            else:
                cur["rows"].append(cells)
        elif line.startswith("- "):
            cur["notes"].append(line[2:].strip())
    return tables


def load_batches(report_dir: Path) -> list:
    index_path = report_dir / "index.json"
    if not index_path.exists():
        raise BuildError(f"missing {index_path}")
    entries = json.loads(index_path.read_text())
    batches = []
    for e in entries:
        folder = report_dir / e["folder"]
        batches.append({
            **e,
            "path": folder,
            "figs": parse_readme(folder / "README.md"),
            "tables": parse_tables(folder / "tables.md"),
        })
    batches.sort(key=lambda b: (b.get("date", ""), b.get("generated", "")), reverse=True)
    return batches


# ---------------------------------------------------------------- queue

def load_queue(queue_path: Path, done_path: Path) -> dict:
    queued = []
    if queue_path.exists():
        for line in queue_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            queued.append(s.split("|", 1)[0].strip())
    events = {}
    order = []
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            when = f"{parts[0]} {parts[1]}"
            kind, label = parts[2], parts[3]
            if kind == "START":
                events[label] = {"label": label, "start": when, "end": None,
                                 "rc": None, "seconds": None}
                order.append(label)
            elif kind == "END" and label in events:
                ev = events[label]
                ev["end"] = when
                for p in parts[4:]:
                    if p.startswith("rc="):
                        ev["rc"] = p[3:]
                    elif p.endswith("s") and p[:-1].isdigit():
                        ev["seconds"] = int(p[:-1])
    running = [events[l] for l in dict.fromkeys(order) if events[l]["end"] is None]
    finished = [events[l] for l in dict.fromkeys(order) if events[l]["end"] is not None]
    finished.sort(key=lambda e: e["end"], reverse=True)
    run_labels = {e["label"] for e in running}
    return {
        "running": running,
        "queued": [q for q in queued if q not in run_labels],
        "finished": finished,
    }


def queue_states(queue: dict, pattern: str) -> list:
    rx = re.compile(pattern)
    rows = []
    for e in queue["running"]:
        if rx.search(e["label"]):
            rows.append((e["label"], "running", f"started {e['start']}"))
    for label in queue["queued"]:
        if rx.search(label):
            rows.append((label, "queued", ""))
    for e in queue["finished"]:
        if rx.search(e["label"]) and not any(r[0] == e["label"] for r in rows):
            state = "done" if e["rc"] == "0" else "failed"
            rows.append((e["label"], state, f"ended {e['end']}, rc={e['rc']}"))
    return rows


# ---------------------------------------------------------------- builder

class Site:
    def __init__(self, content: dict, batches: list, queue: dict, out_dir: Path,
                 link: bool):
        self.content = content
        self.batches = {b["folder"]: b for b in batches}
        self.batch_order = batches
        self.queue = queue
        self.out_dir = out_dir
        self.link = link
        self.files = {}
        self.devices = content.get("site", {}).get("devices", [])

    # ---- figures

    def fig_path(self, batch: str, file: str) -> str:
        b = self.batches.get(batch)
        if b is None:
            raise BuildError(f"unknown batch {batch!r}")
        src = b["path"] / file
        if not src.exists():
            raise BuildError(f"missing figure {src}")
        published = f"figs/{batch}/{file}"
        dest = self.out_dir / published
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        if self.link:
            os.symlink(src.resolve(), dest)
            self.files[published] = str(src.relative_to(ROOT))
        else:
            shutil.copy2(src, dest)
            self.files[published] = str(dest.relative_to(ROOT))
        return published

    def figure(self, blk: dict) -> str:
        batch, file = blk["batch"], blk["file"]
        published = self.fig_path(batch, file)
        meta = self.batches[batch]["figs"].get(file, {})
        caption = blk.get("caption") or meta.get("description", file)
        sources = blk.get("sources") or meta.get("sources", [])
        srcs = ""
        if sources:
            items = "".join(f"<li><code>{esc(s)}</code></li>" for s in sources)
            srcs = (f'<details class="srcs"><summary>Sources ({len(sources)} files)'
                    f"</summary><ul>{items}</ul></details>")
        return (
            f'<figure class="fig"><a href="{esc(published)}" target="_blank" rel="noopener">'
            f'<img src="{esc(published)}" alt="{esc(caption)}" loading="lazy"></a>'
            f'<figcaption><span class="fig-id">{esc(batch)}/{esc(file)}</span> '
            f"{inline(caption)}{srcs}</figcaption></figure>"
        )

    # ---- tables

    @staticmethod
    def _table_html(columns, rows, align=None, caption=None, notes=None,
                    source=None, compact=False) -> str:
        align = align or []
        if not align:
            align = ["left"] + ["right"] * (len(columns) - 1)

        def cls(i):
            a = align[i] if i < len(align) else "left"
            return ' class="num"' if a == "right" else ""

        head = "".join(f"<th{cls(i)}>{inline(c)}</th>" for i, c in enumerate(columns))
        body = "".join(
            "<tr>" + "".join(f"<td{cls(i)}>{inline(c)}</td>" for i, c in enumerate(r)) + "</tr>"
            for r in rows
        )
        cap = f"<caption>{inline(caption)}</caption>" if caption else ""
        tcls = "tbl compact" if compact else "tbl"
        out = f'<div class="tbl-wrap"><table class="{tcls}">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        if notes:
            out += '<ul class="tbl-notes">' + "".join(f"<li>{inline(n)}</li>" for n in notes) + "</ul>"
        out += src_line(source)
        return out

    def table(self, blk: dict) -> str:
        return self._table_html(blk["columns"], blk["rows"], blk.get("align"),
                                blk.get("caption"), blk.get("notes"), blk.get("source"),
                                blk.get("compact", False))

    def batch_table(self, blk: dict) -> str:
        b = self.batches.get(blk["batch"])
        if b is None:
            raise BuildError(f"unknown batch {blk['batch']!r}")
        want = blk["table"]
        matches = [t for t in b["tables"] if t["heading"].startswith(want)]
        if not matches:
            raise BuildError(f"no table starting {want!r} in {blk['batch']}/tables.md")
        t = matches[0]
        cols, rows, align = t["columns"], t["rows"], t["align"]
        if blk.get("rows"):
            prefixes = blk["rows"]
            keep = []
            for p in prefixes:  # keep the order of the prefixes
                keep += [r for r in rows if " | ".join(r).startswith(p) and r not in keep]
            rows = keep
        if blk.get("columns"):
            idx = []
            for c in blk["columns"]:
                if c not in cols:
                    raise BuildError(f"column {c!r} not in table {t['heading']!r}")
                idx.append(cols.index(c))
            cols = [cols[i] for i in idx]
            align = [align[i] for i in idx] if align else []
            rows = [[r[i] for i in idx] for r in rows]
        notes = t["notes"] if blk.get("notes", True) else []
        caption = blk.get("caption", t["heading"])
        source = [f"out/research/report/{blk['batch']}/tables.md"]
        return self._table_html(cols, rows, align, caption, notes, source)

    @staticmethod
    def _cell_record(col: dict, obj, device: str):
        """The per-device result record behind an RRMS cell, for the strict
        failed-curve check; None when the column is not an NGSpice paper RRMS
        or its record cannot be found.  Lookup order: an explicit
        ``record: {dir, method}`` on the column; the value's parent object
        when it carries per_curve / polish history; ``ml_<device>.json`` next
        to the file (swap search); the pipeline records named by a swap
        summary (``pipeline_rrms``) or by a surrogate-gap file name."""
        keys = col["path"].replace("{device}", device).split(".")
        if "rrms" not in keys[-1] or keys[-1].startswith("surrogate"):
            return None

        def from_dir(directory, method=None):
            f = ROOT / directory / f"ml_{device}.json"
            if not f.is_file():
                return None
            methods = json.loads(f.read_text()).get("methods", {})
            if method is None:
                method = (next(iter(methods)) if len(methods) == 1
                          else "surrogate_fd")
            return methods.get(method)

        if col.get("record"):
            return from_dir(col["record"]["dir"], col["record"].get("method"))
        parent = obj
        for key in keys[:-1]:
            parent = parent.get(key) if isinstance(parent, dict) else None
        if isinstance(parent, dict) and any(
                k in parent for k in ("per_curve", "history", "failed_curves")):
            return parent
        if keys[-1] == "pipeline_rrms" and isinstance(obj.get("pipeline"), dict):
            return from_dir(obj["pipeline"].get("dir", ""), "surrogate_fd")
        m = re.match(r"surrogate_gap_(\w+?)_round_(\d+)\.json$", Path(col["file"]).name)
        if m and len(keys) >= 2 and keys[-2] == "pipeline":
            return from_dir(f"out/cold_loop/{m.group(1)}/round_{m.group(2)}/extract",
                            "surrogate_fd")
        if len(keys) == 3 and keys[0] == "devices":
            return from_dir(str(Path(col["file"]).parent))
        return None

    def per_device(self, blk: dict) -> str:
        devices = blk.get("devices") or self.devices
        cols = ["device"] + [c["label"] for c in blk["columns"]]
        fmt = blk.get("format", "{:.4f}")
        data, sources = [], []
        for c in blk["columns"]:
            p = ROOT / c["file"]
            if not p.exists():
                raise BuildError(f"missing {p}")
            data.append(json.loads(p.read_text()))
            if c["file"] not in sources:
                sources.append(c["file"])
        rows, values, status = [], [], []
        for d in devices:
            row, vals, sts = [f"`{d}`"], [], []
            for c, obj in zip(blk["columns"], data):
                v = obj
                for key in c["path"].replace("{device}", d).split("."):
                    v = v.get(key) if isinstance(v, dict) else None
                    if v is None:
                        break
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
                rec = self._cell_record(c, obj, d) if ok else None
                st = None
                if rec is not None:
                    try:
                        st = ss.record_status(rec, d)
                    except KeyError:      # not a frozen-baseline device
                        st = None
                failed = st is not None and st["status"] == ss.FAILED
                row.append("–" if not ok else c.get("format", fmt).format(v)
                           + (" †" if failed else ""))
                vals.append(float(v) if ok else None)
                sts.append(st)
            rows.append(row)
            values.append(vals)
            status.append(sts)
        notes = list(blk.get("notes") or [])
        if blk.get("mean_row", True):
            mean, penal, any_failed = ["mean (all-device)"], ["mean, penalized"], False
            for j, c in enumerate(blk["columns"]):
                col_vals = [v[j] for v in values]
                f = c.get("format", fmt)
                if any(v is None for v in col_vals):
                    mean.append("–")
                    penal.append("–")
                    continue
                bad = [i for i, st in enumerate(status)
                       if st[j] is not None and st[j]["status"] == ss.FAILED]
                if not bad:
                    mean.append(f.format(sum(col_vals) / len(col_vals)))
                    penal.append(mean[-1] if "rrms" in c["path"].rsplit(".", 1)[-1]
                                 else "")
                    continue
                # Never average over missing curves: valid devices only, and
                # the penalized convention on its own row.
                any_failed = True
                good = [v for i, v in enumerate(col_vals) if i not in bad]
                mean.append((f.format(sum(good) / len(good)) if good else "–")
                            + f" ‡ ({len(bad)} failed)")
                pen = [status[i][j]["penalized_rrms"] if i in bad else v
                       for i, v in enumerate(col_vals)]
                penal.append(f.format(sum(pen) / len(pen)))
            rows.append(mean)
            if any_failed:
                rows.append(penal)
                notes.append(
                    "† an included metric curve of this device failed to simulate: the value "
                    "averages only the surviving curves and is not a valid paper RRMS. "
                    "‡ mean over the valid devices only. Penalized row: every device, each "
                    f"failed curve at RRMS {ss.PENALTY:g} (penalized convention, not the paper "
                    "metric; `cryoml.strict_score`).")
        return self._table_html(cols, rows, None, blk.get("caption"), notes or None,
                                sources, compact=True)

    # ---- other blocks

    def queue_block(self, blk: dict) -> str:
        rows = queue_states(self.queue, blk["match"])
        if not rows:
            return '<p class="muted">No matching jobs in the queue or the run log.</p>' + \
                src_line(["out/research/queue.txt", "out/research/done.txt"])
        items = "".join(
            f'<li><span class="chip chip-{st}">{st}</span> <code>{esc(lbl)}</code>'
            f'{" <span class=muted>" + esc(note) + "</span>" if note else ""}</li>'
            for lbl, st, note in rows
        )
        return f'<ul class="qlist">{items}</ul>' + src_line(
            ["out/research/queue.txt", "out/research/done.txt"])

    def timeline(self, blk: dict) -> str:
        items = "".join(
            f'<li><span class="tl-date">{esc(i["date"])}</span>'
            f'<span class="tl-body"><strong>{inline(i["label"])}</strong>'
            f'{(" " + inline(i["note"])) if i.get("note") else ""}</span></li>'
            for i in blk["items"]
        )
        return f'<ol class="timeline">{items}</ol>' + src_line(blk.get("source"))

    def nodes(self, blk: dict) -> str:
        rows = []
        for n in blk["items"]:
            code = "<br>".join(f"<code>{esc(c)}</code>" for c in as_list(n.get("code"))) or \
                f'<span class="muted">{inline(n.get("code_note", "none"))}</span>'
            st = n.get("state", "")
            chip = f'<span class="chip chip-{slug(n.get("chip", "running"))}">{esc(st)}</span>'
            rows.append(
                f"<tr><td><strong>{inline(n['name'])}</strong></td>"
                f"<td>{inline(n.get('method', ''))}</td><td>{code}</td><td>{chip}</td></tr>"
            )
        return ('<div class="tbl-wrap"><table class="tbl nodes"><thead><tr><th>node</th>'
                '<th>method</th><th>code</th><th>status</th></tr></thead><tbody>'
                + "".join(rows) + "</tbody></table></div>" + src_line(blk.get("source")))

    def facts(self, blk: dict) -> str:
        rows = "".join(
            f'<div class="fact"><dt>{inline(i["label"])}</dt><dd>{inline(i["value"])}'
            f'{src_line(i.get("source"))}</dd></div>'
            for i in blk["items"]
        )
        return f'<dl class="facts">{rows}</dl>'

    def block(self, blk) -> str:
        if isinstance(blk, str):
            return f"<p>{inline(blk)}</p>"
        t = blk.get("type", "text")
        if t == "text":
            return f"<p>{inline(blk['text'])}</p>" + src_line(blk.get("source"))
        if t == "code":
            return f'<pre class="code"><code>{esc(blk["text"])}</code></pre>' + src_line(blk.get("source"))
        if t == "list":
            items = "".join(f"<li>{inline(i)}</li>" for i in blk["items"])
            return f"<ul class=\"plain\">{items}</ul>" + src_line(blk.get("source"))
        if t == "figure":
            return self.figure(blk)
        if t == "table":
            return self.table(blk)
        if t == "batch_table":
            return self.batch_table(blk)
        if t == "per_device":
            return self.per_device(blk)
        if t == "queue":
            return self.queue_block(blk)
        if t == "timeline":
            return self.timeline(blk)
        if t == "nodes":
            return self.nodes(blk)
        if t == "facts":
            return self.facts(blk)
        raise BuildError(f"unknown block type {t!r}")

    # ---- claims

    def claim(self, c: dict) -> str:
        status = c.get("status", "in_progress")
        if status not in STATUS_LABEL:
            raise BuildError(f"claim {c.get('id')}: unknown status {status!r}")
        hl = c.get("headline") or {}
        hl_html = ""
        if hl:
            hl_html = (f'<span class="hl"><span class="hl-v">{inline(hl.get("value", ""))}</span>'
                       f' <span class="hl-c">{inline(hl.get("convention", ""))}</span></span>')
        summary = (f'<summary><span class="chip chip-{slug(status)}">{STATUS_LABEL[status]}</span>'
                   f'<span class="stmt">{inline(c["statement"])}</span>{hl_html}</summary>')
        parts = []

        def part(title, inner):
            if inner:
                parts.append(f'<section class="cpart"><h4>{title}</h4>{inner}</section>')

        part("What it is", "".join(self.block(b) for b in as_list(c.get("what"))))
        did = as_list(c.get("did"))
        if did:
            part("What we did", '<ul class="plain">' + "".join(
                f"<li>{inline(d)}</li>" if isinstance(d, str) else f"<li>{self.block(d)}</li>"
                for d in did) + "</ul>")
        res = c.get("result") or {}
        inner = []
        nums = as_list(c.get("numbers"))
        if nums:
            rows = "".join(
                f"<tr><td>{inline(n['label'])}</td><td class=\"num\">{inline(n['value'])}</td>"
                f"<td>{inline(n.get('convention', ''))}</td>"
                f"<td class=\"srccell\">{' '.join('<code>' + esc(s) + '</code>' for s in as_list(n.get('source')))}</td></tr>"
                for n in nums)
            inner.append('<div class="tbl-wrap"><table class="tbl numbers"><thead><tr><th>quantity</th>'
                         '<th class="num">value</th><th>convention</th><th>source</th></tr></thead>'
                         f"<tbody>{rows}</tbody></table></div>")
        for t in as_list(res.get("text")):
            inner.append(self.block(t))
        for f in as_list(res.get("figures")):
            inner.append(self.figure(f))
        for t in as_list(res.get("tables")):
            inner.append(self.block(t) if "type" in t else self.batch_table(t) if "batch" in t else self.table(t))
        for b in as_list(res.get("blocks")):
            inner.append(self.block(b))
        part("Result", "".join(inner))
        cav = as_list(c.get("caveats"))
        if cav:
            part("Caveats and open checks", '<ul class="plain">' + "".join(
                f"<li>{inline(x)}</li>" for x in cav) + "</ul>")
        srcs = as_list(c.get("sources"))
        if srcs:
            part("Source files", '<ul class="srclist">' + "".join(
                f"<li><code>{esc(s)}</code></li>" for s in srcs) + "</ul>")
        meta = f'<p class="cmeta">Claim <code>{esc(c["id"])}</code> · updated {esc(c.get("updated", "–"))}</p>'
        return (f'<details class="claim" id="claim-{esc(c["id"])}">{summary}'
                f'<div class="cbody">{"".join(parts)}{meta}</div></details>')

    # ---- tabs

    def status_block(self, st: dict) -> str:
        if not st:
            return ""

        def items(lst):
            out = []
            for i in lst:
                if isinstance(i, str):
                    out.append(f"<li>{inline(i)}</li>")
                    continue
                q = ""
                if i.get("queue_match"):
                    rows = queue_states(self.queue, i["queue_match"])
                    counts = {}
                    for _, s, _ in rows:
                        counts[s] = counts.get(s, 0) + 1
                    if rows:
                        q = ' <span class="qsum">' + ", ".join(
                            f"{counts[k]} {k}" for k in ("running", "queued", "done", "failed") if k in counts
                        ) + "</span>"
                link = f' <a class="jump" href="#claim-{esc(i["claim"])}">details</a>' if i.get("claim") else ""
                out.append(f"<li>{inline(i['text'])}{q}{link}{src_line(i.get('source'))}</li>")
            return "".join(out)

        rows = [("Question", f"<p>{inline(st['question'])}</p>")]
        if st.get("answer"):
            rows.append(("Current answer", f'<ul class="plain">{items(st["answer"])}</ul>'))
        if st.get("next"):
            rows.append(("Next runs", f'<ul class="plain">{items(st["next"])}</ul>'))
        body = "".join(f'<div class="st-row"><div class="st-k">{k}</div><div class="st-v">{v}</div></div>'
                       for k, v in rows)
        upd = f'<p class="st-upd">Status as of {esc(st["updated"])}</p>' if st.get("updated") else ""
        return f'<section class="status" aria-label="Status">{body}{upd}</section>'

    def section(self, s: dict) -> str:
        sid = s.get("id") or slug(s["title"])
        body = "".join(self.block(b) for b in as_list(s.get("blocks")))
        return f'<section class="sec" id="{esc(sid)}"><h3>{inline(s["title"])}</h3>{body}</section>'

    def paper_panel(self, tab: dict, merged_from: list) -> str:
        claims = [c for c in self.content.get("claims", []) if c.get("tab") == tab["id"]]
        for m in merged_from:
            claims += [c for c in self.content.get("claims", []) if c.get("tab") == m["id"]]
        out = [f'<header class="ph"><h2>{inline(tab["title"])}</h2>']
        if tab.get("subtitle"):
            out.append(f'<p class="ph-sub">{inline(tab["subtitle"])}</p>')
        out.append("</header>")
        if tab.get("state") == "archived":
            out.append(f'<p class="notice">Archived {esc(tab.get("state_date", ""))}. {inline(tab.get("state_note", ""))}</p>')
        if tab.get("may_merge_into"):
            target = tab["may_merge_into"]
            out.append(f'<p class="notice">Status: may merge into the <a href="#{esc(target)}">{esc(self.tab_label(target))}</a> tab.</p>')
        out.append(self.status_block(tab.get("status")))
        for s in as_list(tab.get("sections")):
            if s.get("position") == "before_claims":
                out.append(self.section(s))
        if claims:
            counts = {}
            for c in claims:
                counts[c["status"]] = counts.get(c["status"], 0) + 1
            cnt = " · ".join(f"{counts[k]} {STATUS_LABEL[k]}" for k in STATUS_LABEL if k in counts)
            out.append(f'<section class="claims"><div class="claims-h"><h3>Claims</h3>'
                       f'<span class="muted">{esc(cnt)}</span>'
                       f'<button type="button" class="toggle-all" data-open="0">Expand all</button></div>')
            out.extend(self.claim(c) for c in claims)
            out.append("</section>")
        for s in as_list(tab.get("sections")):
            if s.get("position") != "before_claims":
                out.append(self.section(s))
        for m in merged_from:
            out.append(f'<p class="notice">Merged from the former tab {inline(m["title"])} '
                       f'{esc(m.get("state_date", ""))}.</p>')
            out.append(self.status_block(m.get("status")))
            out.extend(self.section(s) for s in as_list(m.get("sections")))
        return "".join(out)

    def log_panel(self, tab: dict) -> str:
        out = [f'<header class="ph"><h2>{inline(tab["title"])}</h2>']
        if tab.get("subtitle"):
            out.append(f'<p class="ph-sub">{inline(tab["subtitle"])}</p>')
        out.append("</header>")
        out.append('<section class="sec" id="log-queue"><h3>Run queue at build time</h3>')
        q = self.queue
        if q["running"]:
            out.append('<ul class="qlist">' + "".join(
                f'<li><span class="chip chip-running">running</span> <code>{esc(e["label"])}</code> '
                f'<span class="muted">started {esc(e["start"])}</span></li>' for e in q["running"]) + "</ul>")
        else:
            out.append('<p class="muted">No job running.</p>')
        if q["queued"]:
            out.append(f'<p>Queued ({len(q["queued"])}), in run order:</p><ol class="qorder">' + "".join(
                f"<li><code>{esc(l)}</code></li>" for l in q["queued"]) + "</ol>")
        fin = q["finished"][:12]
        if fin:
            rows = [[f"`{e['label']}`", e["end"], e["rc"] or "–",
                     f"{e['seconds']:,}" if e["seconds"] is not None else "–"] for e in fin]
            out.append(self._table_html(["job", "ended", "rc", "seconds"], rows,
                                        ["left", "left", "right", "right"],
                                        f"Last {len(fin)} finished jobs", compact=True))
        out.append(src_line(["out/research/queue.txt", "out/research/done.txt"]))
        out.append("</section>")
        for b in self.batch_order:
            bid = "log-" + slug(b["folder"])
            out.append(f'<section class="sec batch" id="{esc(bid)}"><h3><span class="tl-date">{esc(b["date"])}</span> '
                       f'{esc(b["topic"].replace("_", " "))}</h3>')
            if b.get("summary"):
                out.append('<ul class="plain">' + "".join(f"<li>{inline(s)}</li>" for s in b["summary"]) + "</ul>")
            out.append(src_line([f"out/research/report/{b['folder']}/README.md",
                                 f"out/research/report/{b['folder']}/tables.md"]))
            out.append('<div class="gallery">')
            for png in b.get("pngs", []):
                published = self.fig_path(b["folder"], png)
                desc = b["figs"].get(png, {}).get("description", png)
                out.append(f'<a class="thumb" href="{esc(published)}" target="_blank" rel="noopener">'
                           f'<img src="{esc(published)}" alt="{esc(desc)}" loading="lazy">'
                           f'<span><code>{esc(png)}</code></span></a>')
            out.append("</div>")
            if b["tables"]:
                out.append(f'<details class="srcs"><summary>Tables from tables.md ({len(b["tables"])})</summary>')
                for t in b["tables"]:
                    out.append(self._table_html(t["columns"], t["rows"], t["align"], t["heading"], t["notes"]))
                out.append("</details>")
            gen = b.get("generated", "")
            out.append(f'<p class="cmeta">Generated {esc(gen)} · folder <code>out/research/report/{esc(b["folder"])}/</code></p>')
            out.append("</section>")
        return "".join(out)

    def questions_panel(self, tab: dict) -> str:
        out = [f'<header class="ph"><h2>{inline(tab["title"])}</h2></header>']
        out.append('<section class="sec" id="questions-open"><h3>Open</h3>')
        if tab.get("open"):
            for q in tab["open"]:
                out.append(f'<div class="q"><p><span class="chip chip-open">open</span> <strong>{inline(q["question"])}</strong></p>'
                           f'{"<p>" + inline(q["context"]) + "</p>" if q.get("context") else ""}'
                           f'{src_line(q.get("source"))}</div>')
        else:
            out.append('<p class="muted">No decisions pending.</p>')
        out.append("</section>")
        out.append('<section class="sec" id="questions-decided"><h3>Decided</h3>')
        for q in tab.get("decided", []):
            out.append(f'<div class="q"><p><span class="chip chip-decided">decided</span> <strong>{inline(q["question"])}</strong></p>'
                       f'<p>{inline(q["decision"])}</p>'
                       f'{"<p class=muted>" + inline(q["context"]) + "</p>" if q.get("context") else ""}'
                       f'<p class="cmeta">Recorded {esc(q.get("recorded", "–"))}</p>{src_line(q.get("source"))}</div>')
        out.append("</section>")
        return "".join(out)

    def tab_label(self, tid: str) -> str:
        for t in self.content["tabs"]:
            if t["id"] == tid:
                return t.get("label", tid)
        return tid

    # ---- page

    def render(self) -> str:
        site = self.content.get("site", {})
        tabs = self.content["tabs"]
        ids = {t["id"] for t in tabs}
        for c in self.content.get("claims", []):
            if c.get("tab") not in ids:
                raise BuildError(f"claim {c.get('id')}: unknown tab {c.get('tab')!r}")
        merged = [t for t in tabs if t.get("state") == "merged"]
        aliases = {}
        for m in merged:
            if m.get("merged_into") not in ids:
                raise BuildError(f"tab {m['id']}: merged_into {m.get('merged_into')!r} unknown")
            aliases[m["id"]] = m["merged_into"]
        shown = [t for t in tabs if t.get("state", "active") == "active"] + \
                [t for t in tabs if t.get("state") == "archived"]
        buttons, panels = [], []
        for i, t in enumerate(shown):
            sel = i == 0
            arch = " archived" if t.get("state") == "archived" else ""
            buttons.append(
                f'<button type="button" role="tab" class="tab{arch}" id="tab-{esc(t["id"])}" '
                f'data-tab="{esc(t["id"])}" aria-controls="panel-{esc(t["id"])}" '
                f'aria-selected="{"true" if sel else "false"}" tabindex="{0 if sel else -1}">'
                f'{esc(t.get("label", t["id"]))}</button>'
            )
            kind = t.get("kind", "paper")
            if kind == "log":
                inner = self.log_panel(t)
            elif kind == "questions":
                inner = self.questions_panel(t)
            else:
                inner = self.paper_panel(t, [m for m in merged if m["merged_into"] == t["id"]])
            hidden = "" if sel else " hidden"
            panels.append(f'<section class="panel" role="tabpanel" id="panel-{esc(t["id"])}" '
                          f'aria-labelledby="tab-{esc(t["id"])}"{hidden}>{inner}</section>')
        built = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        conv = "".join(f"<li>{inline(c)}</li>" for c in site.get("conventions", []))
        header = (
            '<header class="top">'
            f'<p class="eyebrow">{inline(site.get("eyebrow", ""))}</p>'
            f'<h1>{esc(site.get("title", "Research results"))}</h1>'
            f'<p class="runhdr">{inline(site.get("run_header", ""))}</p>'
            f'<details class="conv"><summary>Conventions used on every tab</summary><ul class="plain">{conv}</ul></details>'
            f'<p class="built">Built {esc(built)} by <code>scripts/research/build_site.py</code> from '
            '<code>docs/site/content.json</code> and <code>out/research/report/</code></p>'
            "</header>"
        )
        nav = f'<nav class="tabs" role="tablist" aria-label="Paper projects">{"".join(buttons)}</nav>'
        alias_json = json.dumps(aliases).replace("</", "<\\/")
        title = esc(site.get("title", "Research results"))
        return (
            f"<title>{title}</title>\n"
            '<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
            'family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">\n'
            f"<style>{CSS}</style>\n"
            f'<div class="wrap">{header}{nav}<main>{"".join(panels)}</main></div>\n'
            f'<script type="application/json" id="tab-aliases">{alias_json}</script>\n'
            f"<script>{JS}</script>\n"
        )


CSS = r"""
/* Layout: one ledger column; a tab row; each paper tab = status box, then claims as expandable rows, then reference sections. */
:root {
  --bg: #f5f6f6;
  --surface: #ffffff;
  --ink: #161a1c;
  --muted: #59646a;
  --rule: #d6dcdf;
  --rule-strong: #a3adb2;
  --accent: #0b5d78;
  --accent-soft: #e1edf2;
  --ok: #24663a; --ok-bg: #e2f0e5;
  --warn: #7f5200; --warn-bg: #f6ebd3;
  --bad: #9b2c2c; --bad-bg: #f6e1e1;
  --idle: #4f5a60; --idle-bg: #e7eaec;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #111517; --surface: #171c1f; --ink: #e2e7e9; --muted: #98a4aa;
    --rule: #2b3337; --rule-strong: #4c585e; --accent: #6bbfd9; --accent-soft: #15303a;
    --ok: #86c896; --ok-bg: #18301f; --warn: #e3b562; --warn-bg: #352a14;
    --bad: #ec9a9a; --bad-bg: #3a1c1c; --idle: #aab4b9; --idle-bg: #242c30;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #111517; --surface: #171c1f; --ink: #e2e7e9; --muted: #98a4aa;
  --rule: #2b3337; --rule-strong: #4c585e; --accent: #6bbfd9; --accent-soft: #15303a;
  --ok: #86c896; --ok-bg: #18301f; --warn: #e3b562; --warn-bg: #352a14;
  --bad: #ec9a9a; --bad-bg: #3a1c1c; --idle: #aab4b9; --idle-bg: #242c30;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15px; line-height: 1.55; }
img { max-width: 100%; height: auto; display: block; }
.wrap { max-width: 76rem; margin: 0 auto; padding-block: 20px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
a { color: var(--accent); text-underline-offset: 2px; }
code { font-family: var(--font-mono); font-size: 0.86em; overflow-wrap: anywhere; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }
h1, h2, h3, h4 { font-family: var(--font-display); font-weight: 600; text-wrap: balance; line-height: 1.2; margin: 0; }
p { margin: 0; }
.muted { color: var(--muted); }
.top { display: grid; gap: 6px; padding-bottom: 14px; }
.eyebrow { font-family: var(--font-mono); font-size: 12px; color: var(--muted); letter-spacing: 0.04em; }
h1 { font-size: clamp(26px, 4vw, 34px); }
.runhdr { font-family: var(--font-mono); font-size: 12.5px; color: var(--muted); }
.built { font-size: 12px; color: var(--muted); }
.conv summary { cursor: pointer; font-size: 13px; color: var(--accent); }
.conv ul { margin-top: 6px; font-size: 13.5px; max-width: 72ch; }
.tabs { display: flex; flex-wrap: wrap; gap: 2px 4px; border-bottom: 1px solid var(--rule-strong);
  position: sticky; top: env(safe-area-inset-top, 0px); background: var(--bg); z-index: 5; padding-top: 6px; }
.tab { font: 600 14.5px/1.2 var(--font-display); color: var(--muted); background: none; border: 0;
  border-bottom: 3px solid transparent; padding: 9px 10px 8px; cursor: pointer; margin-bottom: -1px; }
.tab:hover { color: var(--ink); }
.tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
.tab.archived { font-style: italic; }
main { padding-top: 22px; }
.panel { display: grid; gap: 26px; min-width: 0; }
.ph { display: grid; gap: 4px; }
.ph h2 { font-size: clamp(21px, 3vw, 26px); }
.ph-sub { color: var(--muted); max-width: 72ch; }
.notice { font-size: 13.5px; padding: 6px 10px; background: var(--accent-soft); border-left: 3px solid var(--accent); max-width: 72ch; }
.status { border: 1px solid var(--rule-strong); background: var(--surface); padding: 4px 16px 10px; display: grid; }
.st-row { display: grid; grid-template-columns: 9.5rem minmax(0, 1fr); gap: 4px 16px; padding: 10px 0; border-bottom: 1px solid var(--rule); }
.st-row:last-of-type { border-bottom: 0; }
.st-k { font-family: var(--font-mono); font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); padding-top: 3px; }
.st-v { min-width: 0; max-width: 78ch; }
.st-upd { font-size: 12px; color: var(--muted); padding-top: 4px; }
ul.plain { margin: 0; padding-left: 1.1em; display: grid; gap: 6px; }
ul.plain li::marker { color: var(--rule-strong); }
.src { font-size: 11.5px; color: var(--muted); margin-top: 3px; line-height: 1.45; }
.src code { font-size: 11px; color: var(--muted); margin-right: 6px; }
.src-k { font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.06em; font-size: 10.5px; margin-right: 4px; }
.qsum { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
.jump { font-size: 12.5px; }
.claims { display: grid; gap: 0; }
.claims-h { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; padding-bottom: 8px; border-bottom: 1px solid var(--rule-strong); }
.claims-h h3 { font-size: 19px; }
.toggle-all { margin-left: auto; font: 500 12.5px var(--font-mono); background: var(--surface); color: var(--ink);
  border: 1px solid var(--rule-strong); padding: 4px 10px; cursor: pointer; }
.toggle-all:hover { border-color: var(--accent); color: var(--accent); }
details.claim { border-bottom: 1px solid var(--rule); }
details.claim > summary { list-style: none; cursor: pointer; display: grid; grid-template-columns: 6.6rem minmax(0, 1fr) auto;
  gap: 4px 14px; align-items: baseline; padding: 11px 4px 11px 0; }
details.claim > summary::-webkit-details-marker { display: none; }
details.claim > summary:hover .stmt { color: var(--accent); }
details.claim > summary::after { content: "+"; font-family: var(--font-mono); color: var(--muted); grid-column: 3; grid-row: 1; justify-self: end; }
details.claim[open] > summary::after { content: "−"; }
.stmt { font-weight: 600; min-width: 0; }
.hl { grid-column: 2; font-size: 13.5px; color: var(--muted); }
.hl-v { font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 500; }
.chip { display: inline-block; font: 500 11px/1.5 var(--font-mono); text-transform: lowercase; letter-spacing: 0.02em;
  padding: 1px 7px; border: 1px solid currentColor; white-space: nowrap; justify-self: start; }
.chip-verified, .chip-done, .chip-decided { color: var(--ok); background: var(--ok-bg); }
.chip-in-progress, .chip-in_progress, .chip-running { color: var(--warn); background: var(--warn-bg); }
.chip-refuted, .chip-failed { color: var(--bad); background: var(--bad-bg); }
.chip-planned, .chip-queued, .chip-open, .chip-built, .chip-smoke { color: var(--idle); background: var(--idle-bg); }
.cbody { padding: 4px 0 20px 0; display: grid; gap: 16px; min-width: 0; }
@media (min-width: 720px) { .cbody { padding-left: calc(6.6rem + 14px); } }
.cpart { display: grid; gap: 8px; min-width: 0; }
.cpart h4 { font-family: var(--font-mono); font-weight: 500; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.cpart > p, .cpart ul.plain { max-width: 78ch; }
.cmeta { font-size: 11.5px; color: var(--muted); }
.srclist { margin: 0; padding-left: 1.1em; font-size: 12.5px; }
.sec { display: grid; gap: 10px; min-width: 0; }
.sec > h3 { font-size: 19px; padding-bottom: 6px; border-bottom: 1px solid var(--rule-strong); }
.sec > p { max-width: 78ch; }
.tbl-wrap { overflow-x: auto; max-width: 100%; -webkit-overflow-scrolling: touch; }
table.tbl { border-collapse: collapse; font-size: 13px; min-width: 100%; width: max-content; max-width: none; font-variant-numeric: tabular-nums; }
table.tbl caption { text-align: left; font-weight: 600; font-size: 13px; padding: 2px 0 6px; caption-side: top; }
table.tbl th { font-weight: 600; text-align: left; border-bottom: 1px solid var(--rule-strong); padding: 5px 10px 5px 0; vertical-align: bottom; }
table.tbl td { border-bottom: 1px solid var(--rule); padding: 4px 10px 4px 0; vertical-align: top; }
table.tbl .num { text-align: right; font-family: var(--font-mono); font-size: 12.5px; white-space: nowrap; }
table.tbl th.num { font-family: var(--font-body); font-size: 13px; }
table.tbl tbody tr:hover td { background: var(--accent-soft); }
table.compact td, table.compact th { padding-top: 2px; padding-bottom: 2px; }
table.numbers td.srccell code { font-size: 11px; color: var(--muted); display: block; }
table.nodes td { min-width: 9rem; max-width: 30rem; white-space: normal; }
.tbl-notes { margin: 6px 0 0; padding-left: 1.1em; font-size: 12.5px; color: var(--muted); max-width: 90ch; }
.fig { margin: 0; display: grid; gap: 6px; max-width: 62rem; }
.fig img { border: 1px solid var(--rule); background: #ffffff; }
.fig figcaption { font-size: 13px; color: var(--muted); max-width: 90ch; }
.fig-id { font-family: var(--font-mono); font-size: 11.5px; color: var(--ink); margin-right: 4px; }
details.srcs summary { cursor: pointer; font-size: 12px; color: var(--accent); margin-top: 4px; }
details.srcs ul { margin: 4px 0 0; padding-left: 1.1em; font-size: 11.5px; }
details.srcs .tbl-wrap { margin-top: 12px; }
.pre, pre.code { margin: 0; padding: 10px 12px; background: var(--surface); border: 1px solid var(--rule); overflow-x: auto; font-size: 12.5px; }
.timeline { list-style: none; margin: 0; padding: 0; display: grid; gap: 0; }
.timeline li { display: grid; grid-template-columns: 8.5rem minmax(0, 1fr); gap: 4px 14px; padding: 8px 0; border-bottom: 1px solid var(--rule); }
.tl-date { font-family: var(--font-mono); font-size: 13px; font-variant-numeric: tabular-nums; color: var(--accent); }
.qlist { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; font-size: 13.5px; }
.qorder { margin: 0; padding-left: 2em; font-size: 13px; columns: 2 18rem; }
.facts { margin: 0; display: grid; gap: 0; }
.fact { display: grid; grid-template-columns: 12rem minmax(0, 1fr); gap: 4px 14px; padding: 7px 0; border-bottom: 1px solid var(--rule); }
.fact dt { font-weight: 600; }
.fact dd { margin: 0; min-width: 0; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 210px), 1fr)); gap: 12px; }
.thumb { display: grid; gap: 4px; text-decoration: none; color: var(--ink); font-size: 12px; }
.thumb img { border: 1px solid var(--rule); background: #ffffff; aspect-ratio: 4 / 3; object-fit: contain; width: 100%; }
.thumb:hover img { border-color: var(--accent); }
.q { display: grid; gap: 4px; padding: 10px 0; border-bottom: 1px solid var(--rule); max-width: 80ch; }
@media (max-width: 640px) {
  .st-row, .timeline li, .fact { grid-template-columns: minmax(0, 1fr); }
  details.claim > summary { grid-template-columns: minmax(0, 1fr) auto; }
  details.claim > summary .chip { grid-column: 1; grid-row: 1; }
  details.claim > summary::after { grid-column: 2; grid-row: 1; }
  .stmt, .hl { grid-column: 1 / -1; }
}
@media (prefers-reduced-motion: no-preference) { .tab, .thumb img { transition: color .12s, border-color .12s; } }
"""

JS = r"""
(function () {
  var KEY = "cryoml-site-tab";
  var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));
  var ids = tabs.map(function (t) { return t.getAttribute("data-tab"); });
  var aliases = {};
  try { aliases = JSON.parse(document.getElementById("tab-aliases").textContent) || {}; } catch (e) {}
  function panelOf(el) { while (el && !(el.getAttribute && el.getAttribute("role") === "tabpanel")) el = el.parentNode; return el; }
  function resolve(token) {
    if (!token) return null;
    token = String(token).replace(/^#/, "");
    if (aliases[token]) token = aliases[token];
    if (ids.indexOf(token) >= 0) return { tab: token, target: null };
    var el = document.getElementById(token);
    var p = el && panelOf(el);
    if (p) return { tab: p.id.replace(/^panel-/, ""), target: el };
    return null;
  }
  function show(id, setHash) {
    tabs.forEach(function (t) {
      var on = t.getAttribute("data-tab") === id;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      var p = document.getElementById("panel-" + t.getAttribute("data-tab"));
      if (p) p.hidden = !on;
    });
    try { localStorage.setItem(KEY, id); } catch (e) {}
    if (setHash) {
      try { history.replaceState(null, "", "#" + id); } catch (e) { try { location.hash = id; } catch (e2) {} }
    }
  }
  function reveal(target) {
    if (!target) return;
    var d = target;
    while (d && d.tagName) { if (d.tagName === "DETAILS") d.open = true; d = d.parentNode; }
    try { target.scrollIntoView({ block: "start" }); } catch (e) {}
  }
  function fromHash() {
    var r = resolve(location.hash);
    if (r) { show(r.tab, false); reveal(r.target); return true; }
    return false;
  }
  if (!fromHash()) {
    var stored = null;
    try { stored = localStorage.getItem(KEY); } catch (e) {}
    var r = resolve(stored);
    show(r ? r.tab : ids[0], false);
  }
  window.addEventListener("hashchange", fromHash);
  tabs.forEach(function (t, i) {
    t.addEventListener("click", function () { show(t.getAttribute("data-tab"), true); });
    t.addEventListener("keydown", function (ev) {
      var j = null;
      if (ev.key === "ArrowRight") j = (i + 1) % tabs.length;
      else if (ev.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
      else if (ev.key === "Home") j = 0;
      else if (ev.key === "End") j = tabs.length - 1;
      if (j === null) return;
      ev.preventDefault();
      tabs[j].focus();
      show(tabs[j].getAttribute("data-tab"), true);
    });
  });
  document.addEventListener("click", function (ev) {
    var a = ev.target.closest ? ev.target.closest('a[href^="#"]') : null;
    if (!a) return;
    var r = resolve(a.getAttribute("href"));
    if (!r) return;
    ev.preventDefault();
    show(r.tab, true);
    reveal(r.target);
    if (!r.target) { try { window.scrollTo(0, 0); } catch (e) {} }
  });
  Array.prototype.slice.call(document.querySelectorAll(".toggle-all")).forEach(function (b) {
    b.addEventListener("click", function () {
      var open = b.getAttribute("data-open") !== "1";
      var sec = b.closest(".claims");
      Array.prototype.slice.call(sec.querySelectorAll("details.claim")).forEach(function (d) { d.open = open; });
      b.setAttribute("data-open", open ? "1" : "0");
      b.textContent = open ? "Collapse all" : "Expand all";
    });
  });
})();
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--content", default=str(ROOT / "docs/site/content.json"))
    ap.add_argument("--report", default=str(ROOT / "out/research/report"))
    ap.add_argument("--out", default=str(ROOT / "out/research/site"))
    ap.add_argument("--queue", default=str(ROOT / "out/research/queue.txt"))
    ap.add_argument("--done", default=str(ROOT / "out/research/done.txt"))
    ap.add_argument("--link", action="store_true", help="symlink figures instead of copying")
    args = ap.parse_args()

    content = json.loads(Path(args.content).read_text())
    batches = load_batches(Path(args.report))
    queue = load_queue(Path(args.queue), Path(args.done))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    site = Site(content, batches, queue, out_dir, args.link)
    try:
        page = site.render()
    except BuildError as e:
        print(f"build_site: {e}", file=sys.stderr)
        return 1
    (out_dir / "index.html").write_text(page)
    files = dict(sorted(site.files.items()))
    (out_dir / "files.json").write_text(json.dumps(files, indent=1) + "\n")
    print(json.dumps(files, indent=1))
    print(f"wrote {out_dir / 'index.html'} ({len(page):,} bytes, {len(files)} images)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
