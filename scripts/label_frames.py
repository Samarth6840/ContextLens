"""Hand-label ContextLens video frames: LOGO boxes, or an explicit NO LOGO.

The gap this fills
------------------
scripts/video_label_sheet.py can only apply a review as NEGATIVES: it writes an
empty label file per index. There is no way to record a logo box, so the loop the
detector work needs - frame -> boxes or NO LOGO -> split by video -> benchmark -
cannot be run at all, and the one open question stays open forever:

    328 / 641 frames get zero detections from the OpenLogo checkpoint.
    A. genuinely no logo      B. missed logo

Answering that requires reviewed ground truth on real frames. Nothing else in the
repo can substitute for it: LogoDet is a nested scene-text tree (520 usable
boxes, scripts/sanitize_dataset.py), and the existing synthetic harness in
fusion_eval/ composites logos onto noise, so it measures compositing, not
detection. So: a labelling tool. stdlib only, no dependency, no build step.

Loop
----
    python scripts/label_frames.py --frames benchmark/eval_video/images \
        --out benchmark/eval_video/labels.jsonl \
        --propose runs/detect/weights/logo_corpus/train/weights/best.pt
    # open the printed URL, label, ctrl-C
    python scripts/label_frames.py --frames ... --out ... \
        --export benchmark/video_labelled

Per frame, one of three verdicts, and NO LOGO is a first-class answer:

    LOGO      boxes confirmed or drawn   -> positive
    NO LOGO   zero boxes, asserted free  -> the only source of true negatives
    SKIP      unlabelled                 -> never exported, never a negative

A frame that is skipped is NOT a negative. An empty label file is an assertion
that the whole frame is background, so an unreviewed frame must never become one.

Model pre-draw
--------------
--propose runs a checkpoint once and caches boxes for the reviewer to confirm,
correct or delete. That is a review accelerator, not ground truth: every box
carries src="model", every drawn box src="human", and the export reports both
counts. Frames whose boxes are all src="model" must not be used to grade the
model that proposed them - their recall is circular. The export writes them to a
separate list so eval can exclude them.

Round-robin, not sequential
---------------------------
Frames are served round-robin across source videos. Labelling 200 consecutive
frames of one clip produces 200 labels of one scene, and split-by-video then
throws 150 of them away because the whole video lands in one split.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAGE = """<!doctype html><meta charset=utf-8><title>label frames</title>
<style>
body{font:13px Inter,-apple-system,system-ui,sans-serif;margin:0;
     background:#F6F4EF;color:#1B1B18;
     display:flex;flex-direction:column;height:100vh}
#bar{padding:6px 10px;background:#fff;border-bottom:1px solid rgba(27,27,24,.16);
     display:flex;gap:14px;align-items:center;flex-wrap:wrap}
#wrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto}
#stage{position:relative;line-height:0}
#stage img{max-width:100%;max-height:100%;object-fit:contain;cursor:crosshair;display:block}
#stage canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#b{background:#fff;border-top:1px solid rgba(27,27,24,.16);
   padding:8px 10px;display:flex;gap:10px;align-items:center}
button{font:inherit;padding:5px 12px;border:0;border-radius:5px;cursor:pointer}
.p{background:#0B7A4F;color:#fff}.f{background:#B33A2D;color:#fff}
.s{background:#EFEDE7;color:#4B4B44}
#br{font:inherit;padding:5px 8px;border:1px solid rgba(27,27,24,.26);
    border-radius:5px;background:#fff;color:#1B1B18}
#st{color:#0B7A4F;font-family:'JetBrains Mono',ui-monospace,monospace}
</style>
<div id=bar>
  <b id=id>-</b><span id=prog>-</span><span id=st>-</span>
  <span style=color:#6E6E6A>drag=draw &middot; d=delete last box &middot;
  <b>l</b>=LOGO &middot; <b>k</b>=NO LOGO &middot; <b>s</b>=SKIP &middot;
  <b>n</b>=next, unlabelled</span>
</div>
<!-- The canvas must be a sibling of the img, not the img itself: an <img> has
     no 2D context, so drawing straight onto it throws and kills load(). -->
<div id=wrap><div id=stage><img id=im><canvas id=ov></canvas></div></div>
<div id=b>
  <button class=p onclick=save('logo')>LOGO (<span id=n>0</span>)</button>
  <button class=f onclick=save('free')>NO LOGO</button>
  <button class=s onclick=save('skip')>SKIP</button>
  <input id=br placeholder="brand (optional, Enter to commit - applies to every box in this frame)" size=44>
  <button onclick="location.reload()">&gt; reload</button>
</div>
<script>
let cur=null, boxes=[], nat=[0,0], cv=null;
const $=s=>document.querySelector(s);
async function load(){
  const r=await fetch('/api/next'); cur=await r.json();
  if(!cur){$('#id').textContent='DONE - every frame reviewed or skipped';return}
  boxes=(cur.proposals||[]).map(b=>({...b,src:'model',on:true}));
  $('#im').src='/img/'+cur.file;
  $('#id').textContent=cur.video+'#'+cur.frame_index;
  await new Promise(res=>{const i=$('#im');i.complete?res():i.onload=res});
  nat=[$('#im').naturalWidth,$('#im').naturalHeight];
  cv=$('#im').getBoundingClientRect();
  // The overlay must match the *displayed* image, not its natural size, or the
  // boxes land in the wrong place the moment the frame is scaled to fit.
  const ov=$('#ov'); ov.width=Math.round(cv.width); ov.height=Math.round(cv.height);
  $('#br').value=''; $('#br').blur(); draw();
  const p=await (await fetch('/api/state')).json();
  // Show judged-vs-total plus the verdict split. "200 labelled" is not the
  // goal: 200 frames that RESOLVED a logo or confirmed its absence. SKIP does
  // not count, and the logo count is what recall and AUROC are blocked on.
  $('#prog').textContent=`${p.judged} judged · ${p.logo} logo · ${p.free} free · ${p.skip} skip`;
  $('#st').textContent=p.mix;
}
function sc(){return [cv.width/nat[0],cv.height/nat[1]]}
function draw(){
  const c=$('#ov'),ctx=c.getContext('2d'),s=sc();
  // Live box count on the LOGO button. This span used to be hardcoded 0 and
  // nothing ever wrote to it, so the button read "LOGO (0)" forever and it
  // looked like drawing boxes was not registering.
  $('#n').textContent=boxes.length;
  ctx.clearRect(0,0,c.width,c.height);
  ctx.lineWidth=2;
  boxes.forEach((b,i)=>{
    ctx.strokeStyle=b.on?(b.src==='model'?'#3af':'#fa3'):'#666';
    ctx.fillStyle=b.on?(b.src==='model'?'rgba(50,170,255,.18)':'rgba(255,170,50,.18)'):'transparent';
    const x=b.x*s[0],y=b.y*s[1],w=b.w*s[0],h=b.h*s[1];
    if(!b.on)ctx.setLineDash([5,4]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
    ctx.fillRect(x,y,w,h);
    ctx.font='11px monospace';
    // Index text sits on unknown video, so it carries a dark halo rather
    // than picking one colour: white-on-bright and black-on-dark both fail.
    ctx.lineWidth=3;ctx.strokeStyle='rgba(0,0,0,.75)';
    ctx.strokeText(`${i}${b.src==='model'?'m':'h'}`,x+2,y+12);
    ctx.fillStyle=b.on?'#fff':'#bbb';ctx.fillText(`${i}${b.src==='model'?'m':'h'}`,x+2,y+12);
    ctx.lineWidth=2;
  });
  if(drag){const a=Math.min(drag.x0,drag.x1),b2=Math.min(drag.y0,drag.y1);
    ctx.setLineDash([4,3]);ctx.strokeStyle='#fa3';
    ctx.strokeRect(a*s[0],b2*s[1],Math.abs(drag.x1-drag.x0)*s[0],Math.abs(drag.y1-drag.y0)*s[1]);
    ctx.setLineDash([])}
}
let drag=null;
$('#im').addEventListener('mousedown',e=>{
  const [x,y]=xy(e);drag={x0:x,y0:y,x1:x,y1:y};draw()});
addEventListener('mousemove',e=>{if(drag){const[x,y]=xy(e);drag.x1=x;drag.y1=y;draw()}});
addEventListener('mouseup',()=>{
  if(!drag)return;const d=drag;drag=null;
  const x=Math.min(d.x0,d.x1),y=Math.min(d.y0,d.y1);
  const w=Math.abs(d.x1-d.x0),h=Math.abs(d.y1-d.y0);
  if(w*h>nat[0]*nat[1]*0.25){alert('box covers >25% of the frame - is that a logo, or a scene/ad/card?');return}
  if(w>=4&&h>=4)boxes.push({x,y,w,h,src:'human',on:true});
  draw()});
$('#im').addEventListener('click',e=>{
  const [x,y]=xy(e),hit=boxes.findIndex(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);
  if(hit>=0){boxes[hit].on=!boxes[hit].on;draw()}});
function xy(e){const r=$('#im').getBoundingClientRect(),s=sc();
  return[(e.clientX-r.left)/s[0],(e.clientY-r.top)/s[1]]}
async function save(verdict){
  const keep=boxes.filter(b=>b.on);
  if(verdict==='logo'&&!keep.length){
    // The backslash-n sequences are doubled: this JS lives in a Python string,
    // so a single one would become a real newline and fail to parse.
    alert('Draw at least one box first.\\n\\nDrag on the frame to draw a box.\\nIf there is no logo, press NO LOGO (k) instead.\\nIf you cannot judge the frame, press SKIP (s).');
    return}
  // One brand per frame. A frame usually carries one mark repeated on product,
  // endcard and overlay; where it does not, leave it blank rather than guess.
  const br=$('#br').value.trim();
  if(verdict==='logo'&&br)keep.forEach(b=>b.brand=br);
  await fetch('/api/label',{method:'POST',body:JSON.stringify(
    {file:cur.file,video:cur.video,frame_index:cur.frame_index,
     verdict,boxes:verdict==='logo'?keep:[]})});
  await load()}
async function skip(){  // n — move on WITHOUT recording a verdict
  if(!cur)return;
  await fetch('/api/defer',{method:'POST',body:JSON.stringify({file:cur.file})});
  await load()}
// Enter/Tab commits the brand box and hands focus back to the page. Without this
// the brand input keeps focus once clicked, and the guard below then swallows
// every verdict key: the frame stops advancing and nothing is saved, with no
// visible error. Typing "Rolex" must not fire LOGO on the `l`, so the guard
// stays; the input has to give focus back explicitly instead.
$('#br').addEventListener('keydown',e=>{
  if(e.key==='Enter'||e.key==='Tab'){$('#br').blur();}
});
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;   // typing a brand must not trigger keys
  if(e.key==='d'){boxes.pop();draw()}
  else if(e.key==='k')save('free');else if(e.key==='n')skip();
  else if(e.key==='l')save('logo');else if(e.key==='s')save('skip')});
load();
</script>"""


# ---------------------------------------------------------------- frame pool

def video_of(stem: str) -> str:
    """`<video_md5>_<source_frame_index>` -> video id. The index is the identity."""
    return stem.rsplit("_", 1)[0]


def frame_index_of(stem: str) -> int:
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def pool(frames_root: Path) -> list[Path]:
    return sorted(p for p in frames_root.rglob("*.jpg") if p.is_file())


def propose(weights: str, files: list[Path], conf: float, imgsz: int,
            device: str | None) -> dict[str, list[dict]]:
    """Cache detector boxes for review. A proposal is a starting point the
    reviewer confirms, corrects or deletes - never ground truth."""
    if not weights:
        return {}
    from ultralytics import YOLO
    model = YOLO(weights)
    out: dict[str, list[dict]] = {}
    for i, f in enumerate(files):
        r = model.predict(str(f), imgsz=imgsz, conf=conf, device=device,
                          verbose=False)[0]
        b = r.boxes
        xy = b.xyxy.cpu().numpy() if b is not None and len(b) else np.zeros((0, 4))
        cf = b.conf.cpu().numpy() if b is not None and len(b) else np.zeros((0,))
        out[f.name] = [{"x": float(a), "y": float(bb), "w": float(c - a),
                        "h": float(d - bb), "conf": round(float(v), 3)}
                       for (a, bb, c, d), v in zip(xy, cf)]
        if (i + 1) % 200 == 0:
            print(f"  proposed {i+1}/{len(files)}", flush=True)
    return out


# ---------------------------------------------------------------- the store

class Store:
    """label rows, keyed by frame file, in the order they were reviewed.

    Row is the whole ground-truth record: verdict, pixel boxes, and per-box
    provenance. `src` decides whether a frame is usable for grading the model
    that proposed it, so it is written at label time, not reconstructed later.
    """

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["file"]] = r

    def put(self, row: dict) -> None:
        self.rows[row["file"]] = row
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def labelled(self) -> set[str]:
        return set(self.rows)

    def next_frame(self, files: list[Path], done: set[str],
                   deferred: set[str] | None = None) -> Path | None:
        """Round-robin across videos, then in timeline order within a video.

        Sequential order hands the reviewer 200 frames of one clip, and
        split-by-video then discards all but one video's worth of them.

        `deferred` is a session-local set of frames the reviewer jumped past
        with `n`. They are skipped for now but stay unlabelled, so they come
        back once the easy frames are done. They are deliberately NOT stored
        as SKIP: SKIP means unjudgeable, and borrowing it for "look at this
        later" would quietly drain frames out of the 200.
        """
        deferred = deferred or set()
        by_video: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            if f.name not in done and f.name not in deferred:
                by_video[video_of(f.stem)].append(f)
        if not by_video:
            return None
        vids = sorted(by_video)
        return by_video[vids[len(done) % len(vids)]][0]


# ---------------------------------------------------------------- http

def serve(args: argparse.Namespace, files: list[Path],
          props: dict[str, list[dict]], store: Store) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # Frames jumped past with `n`. Session-local and never written to the store,
    # so deferring a hard frame cannot cost you a slot in the 200.
    deferred: set[str] = set()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # one line per keystroke is not a log
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html")
            if p == "/api/next":
                f = store.next_frame(files, store.labelled(), deferred)
                if f is None:
                    return self._send(200, b'null', "application/json")
                return self._send(200, json.dumps({
                    "file": f.name, "video": video_of(f.stem),
                    "frame_index": frame_index_of(f.stem),
                    "proposals": props.get(f.name, []),
                }).encode(), "application/json")
            if p == "/api/state":
                rows = list(store.rows.values())
                mix: dict[str, int] = defaultdict(int)
                for r in rows:
                    mix[r["video"]] += 1
                # Verdict split, because `done` alone hides the number that
                # actually gates the audit: logo-frame recall and AUROC are
                # unmeasurable without positives, and a wall of SKIP or free
                # frames can reach 200 while both stay undefined.
                verdicts: dict[str, int] = defaultdict(int)
                for r in rows:
                    verdicts[r["verdict"]] += 1
                return self._send(200, json.dumps({
                    "done": len(rows), "total": len(files),
                    "judged": verdicts["logo"] + verdicts["free"],
                    "logo": verdicts["logo"], "free": verdicts["free"],
                    "skip": verdicts["skip"],
                    "mix": " ".join(f"{k}:{v}" for k, v in sorted(mix.items())),
                }).encode(), "application/json")
            if p.startswith("/img/"):
                f = next((x for x in files if x.name == p[5:]), None)
                if f is None or not f.is_file():
                    return self._send(404, b"no", "text/plain")
                ctype = mimetypes.guess_type(f.name)[0] or "image/jpeg"
                return self._send(200, f.read_bytes(), ctype)
            self._send(404, b"no", "text/plain")

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) or b"{}"
            if self.path.split("?")[0] == "/api/defer":
                # `n` — jump to the next frame without recording a verdict.
                row = json.loads(body or b"{}")
                if row.get("file"):
                    deferred.add(row["file"])
                    print(f"  {row['file']}  defer  (not labelled, not counted)",
                          flush=True)
                return self._send(200, b"ok", "text/plain")
            row = json.loads(body or b"{}")
            need = ("file", "video", "frame_index", "verdict", "boxes")
            if any(k not in row for k in need):
                return self._send(400, b"bad row", "text/plain")
            if row["verdict"] not in ("logo", "free", "skip"):
                return self._send(400, b"bad verdict", "text/plain")
            row["human_drawn"] = any(b.get("src") == "human" for b in row["boxes"])
            store.put(row)
            deferred.discard(row["file"])
            mix: dict[str, int] = defaultdict(int)
            for r in store.rows.values():
                if r["verdict"] != "skip":
                    mix[r["video"]] += 1
            print(f"  {row['file']}  {row['verdict']:4s} "
                  f"boxes={len(row['boxes'])}  total={len(store.rows)}  "
                  f"{dict(sorted(mix.items()))}", flush=True)
            self._send(200, b"ok", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"label {len(files)} frames -> http://127.0.0.1:{args.port}/"
          f"   ({len(store.rows)} already done)")
    print("ctrl-C to stop; rerun the same command to resume.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {len(store.rows)} frames reviewed -> {args.out}")
    return 0


# ---------------------------------------------------------------- export

def export(args: argparse.Namespace, files: list[Path], store: Store) -> int:
    """YOLO dataset split BY VIDEO, plus the accounting the user asked for:
    how many frames really contain logos, how small are the logos, and how many
    frames are asserted logo-free (the FP/logo-free-frame denominator)."""
    import cv2
    import importlib.util as ilu
    _sp = ilu.spec_from_file_location(
        "evf", Path(__file__).resolve().parent / "extract_video_frames.py")
    _evf = ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_evf)
    split_order = _evf.split_order

    out = Path(args.export)
    rows = [r for r in store.rows.values() if r["verdict"] != "skip"]
    if not rows:
        sys.exit("FATAL: nothing to export (no frames labelled non-skip yet)")

    videos = sorted({r["video"] for r in rows})
    order = split_order(len(videos))
    vid_split = {v: order[i % len(order)] for i, v in enumerate(videos)}
    src = {f.name: f for f in files}

    areas, aspects, per_split, logo_free, model_only = [], [], defaultdict(int), [], []
    for r in rows:
        sp = vid_split[r["video"]]
        d, dl = out / "images" / sp, out / "labels" / sp
        d.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        f = src.get(r["file"])
        if f is None or not f.is_file():
            print(f"  WARN missing frame, skipped: {r['file']}")
            continue
        cv2.imwrite(str(d / r["file"]), cv2.imread(str(f)))
        if not r["boxes"]:
            (dl / f"{Path(r['file']).stem}.txt").write_text("")
            logo_free.append({"file": r["file"], "video": r["video"], "split": sp})
            per_split[sp] += 1
            continue
        h, w = cv2.imread(str(f)).shape[:2]
        lines = []
        for b in r["boxes"]:
            x, y, bw, bh = b["x"], b["y"], b["w"], b["h"]
            if bw < 1 or bh < 1:
                continue
            lines.append(f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} "
                         f"{bw/w:.6f} {bh/h:.6f}")
            areas.append(bw * bh / (w * h))
            aspects.append(bw / max(bh, 1e-6))
        (dl / f"{Path(r['file']).stem}.txt").write_text("\n".join(lines) + "\n")
        per_split[sp] += 1
        if not r.get("human_drawn"):
            model_only.append({"file": r["file"], "video": r["video"],
                               "frame_index": r["frame_index"], "split": sp,
                               "boxes": len(lines)})

    pos_rows = [r for r in rows if r["boxes"]]
    a = np.array(areas) if areas else np.zeros(0)
    ar = np.array(aspects) if aspects else np.zeros(0)
    # Ultralytics hard-errors without BOTH train: and val: keys. With one video
    # only one split exists, so the missing one points at a present split and the
    # report says so - a val set aliased onto train is a fact to know, not to hide.
    present = [s for s in ("train", "val", "test") if (out / "images" / s).is_dir()]
    alias = ""
    keys = []
    for s in ("train", "val"):
        if s in present:
            keys.append(f"{s}: images/{s}")
        elif present:
            keys.append(f"{s}: images/{present[0]}")
            alias += f"{s}->{present[0]} "
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n# hand-reviewed video frames, split by video\n"
        + "".join(k + "\n" for k in keys)
        + "\nnc: 1\nnames: ['logo']\n")
    (out / "logo_free.json").write_text(json.dumps(logo_free, indent=1))
    (out / "model_prelabelled.json").write_text(json.dumps(model_only, indent=1))
    # Reviewed brand names, carried beside the YOLO labels because YOLO has
    # nowhere to put them. The detector stays single-class; this is the seed for
    # the identity stage, not a training target.
    brands = {r["file"]: [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                           "brand": b.get("brand", ""),
                           "src": b.get("src", "human")}
                          for b in r["boxes"] if b.get("brand")]
              for r in rows if any(b.get("brand") for b in r["boxes"])}
    if brands:
        (out / "brands.json").write_text(json.dumps(brands, indent=1))
    q = lambda p: (None if not len(a) else round(float(np.percentile(a, p)), 5))
    rep = {
        "frames_reviewed": len(store.rows),
        "frames_exported": sum(per_split.values()),
        "split_frames": dict(per_split),
        "videos": {v: vid_split[v] for v in videos},
        "logo_free_frames": len(logo_free),
        "logo_free_per_split": dict(Counter(x["split"] for x in logo_free)),
        "logo_frames": len(pos_rows),
        "boxes": int(a.size),
        "boxes_per_positive_frame": round(
            a.size / max(1, len(pos_rows)), 2),
        "frac_frames_with_a_logo": round(
            len(pos_rows) / max(1, sum(per_split.values())), 4),
        "area_frac_p10": q(10), "area_frac_p50": q(50), "area_frac_p90": q(90),
        "frac_boxes_under_1pct_frame": round(float((a < 0.01).mean()), 4) if len(a) else None,
        "aspect_p50": round(float(np.percentile(ar, 50)), 2) if len(ar) else None,
        # NO LOGO frames are always human verdicts, so they are counted apart and
        # are never "model prelabelled" - a frame the model found nothing in and
        # that was then called logo-free is circular in the worst direction.
        "logo_frames_model_prelabelled": len(model_only),
        "logo_frames_human_drawn": len(pos_rows) - len(model_only),
        "frames_with_named_brand": len(brands),
        "named_brands": sorted({b["brand"] for v in brands.values() for b in v}),
    }
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\n-> {out}\nlogo-free index  -> {out/'logo_free.json'}")
    if model_only:
        print(f"CIRCULAR: {len(model_only)} logo frames came from model proposals. "
              f"Excluded when grading the model that proposed them -> "
              f"{out/'model_prelabelled.json'}")
    else:
        print("circularity: none. Every box is human, so this set grades ANY "
              "checkpoint and survives retraining.")
    if len(videos) < 3:
        print(f"WARNING: {len(videos)} video(s). Below 3, split_order gives a 2-way "
              f"holdout and val==test. Need 3+ distinct videos to tune on.")
    if alias:
        print(f"WARNING: data.yaml aliases {alias.strip()} - Ultralytics needs both "
              f"keys. Scores from that split are not held out.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="benchmark/eval_video/images")
    ap.add_argument("--out", default="benchmark/eval_video/labels.jsonl")
    ap.add_argument("--propose", default="", help="weights to pre-draw boxes with")
    ap.add_argument("--propose-conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--only-video", default="",
                    help="restrict the queue to one video md5 prefix, so labeling "
                         "effort goes to the videos that move the audit")
    ap.add_argument("--export", default="", help="write the YOLO dataset and exit")
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse a cached --propose run instead of recomputing. "
                         "Off by default: a cache must never reappear on its own, "
                         "or a zero-circularity run silently becomes a prelabelled "
                         "one and every verdict stops being a human verdict")
    args = ap.parse_args()

    files = pool(Path(args.frames))
    if args.only_video:
        # Pin the queue to one source video. 615 unlabelled frames is far more
        # human time than the audit needs, and most of it sits in near-duplicate
        # logo-free scenes that move no metric. Labeling effort should go where
        # the metric is blocked: positive logo frames.
        files = [f for f in files if f.name.split("_")[0] == args.only_video]
        if not files:
            sys.exit(f"FATAL: no frames for video {args.only_video} under {args.frames}")
        print(f"pinned to {args.only_video}: {len(files)} frames")
    if not files:
        sys.exit(f"FATAL: no frames under {args.frames}")
    store = Store(Path(args.out))
    if args.export:
        return export(args, files, store)

    cache = Path(args.out).with_suffix(".proposals.json")
    props: dict[str, list[dict]] = {}
    if args.propose:
        print(f"proposing with {args.propose} ...", flush=True)
        props = propose(args.propose, files, args.propose_conf, args.imgsz,
                        args.device)
        cache.write_text(json.dumps(props))
        n = sum(len(v) for v in props.values())
        print(f"  {n} proposals over {len(files)} frames "
              f"({n/max(1,len(files)):.1f}/frame) -> {cache}")
    elif args.use_cache and cache.exists():
        props = json.loads(cache.read_text())
        print(f"reusing {cache}: {sum(len(v) for v in props.values())} proposals")
    else:
        stale = " (a cache exists - pass --use-cache to load it, or --propose to rebuild)" \
            if cache.exists() else ""
        print(f"ZERO-CIRCULARITY MODE{stale}", flush=True)
        print("  no model pre-draw. Every box drawn and every verdict recorded is a")
        print("  HUMAN judgement, so this ground truth grades any checkpoint and")
        print("  stays valid after the model is retrained.")
    return serve(args, files, props, store)


if __name__ == "__main__":
    sys.exit(main())
