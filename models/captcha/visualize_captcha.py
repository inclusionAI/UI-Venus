#!/usr/bin/env python3
# coding: utf-8
"""Visualize one result produced by infer_captcha.py.

Usage:
    python models/captcha/visualize_captcha.py --result results/captcha/result.json --output results/captcha/result.html

This module is deliberately separate from inference. It reads one JSON result
and creates a self-contained HTML file for parsed Click, LongPress, Type, and
Drag actions. Coordinate-bearing actions are drawn over the source image;
Type actions are also listed explicitly because they do not carry coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Sequence

from infer_captcha import image_to_data_url, parse_actions, read_image_size


def denormalize(
    x: float,
    y: float,
    width: float,
    height: float,
    coord_scale: float,
) -> tuple[float, float]:
    if coord_scale <= 0:
        return float(x), float(y)
    return (
        float(x) / coord_scale * width,
        float(y) / coord_scale * height,
    )


def load_result(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise ValueError("结果文件必须是单个 JSON 对象: %s" % path)
    return result


def _resolve_image(
    result: dict[str, Any],
    result_path: str,
    image_override: Optional[str],
) -> str:
    value = image_override if image_override is not None else result.get("image")
    if not value:
        raise ValueError("结果中没有 image；请用 --image 指定原图")
    value = os.path.expanduser(str(value))
    if os.path.isabs(value):
        candidates = [value]
    elif image_override is not None:
        # Resolve command-line override paths from the current working directory.
        candidates = [value]
    else:
        # Resolve result image paths from the result JSON directory.
        candidates = [os.path.join(os.path.dirname(result_path), value)]
    for candidate in candidates:
        absolute = os.path.abspath(candidate)
        if os.path.isfile(absolute):
            return absolute
    raise FileNotFoundError("找不到可视化原图: %s" % value)


def build_visual_payload(
    result: dict[str, Any],
    result_path: str,
    image_override: Optional[str] = None,
    coord_scale_override: Optional[float] = None,
    path_base: Optional[str] = None,
) -> dict[str, Any]:
    image_path = _resolve_image(result, result_path, image_override)
    relative_base = os.path.abspath(
        path_base or os.path.dirname(os.path.abspath(result_path))
    )
    image_size = result.get("image_size")
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) < 2
    ):
        image_size = read_image_size(image_path)
    width, height = float(image_size[0]), float(image_size[1])
    coord_scale = (
        float(coord_scale_override)
        if coord_scale_override is not None
        else float(result.get("coord_scale", 999.0))
    )
    actions = result.get("parsed_actions")
    if not isinstance(actions, list):
        actions = parse_actions(result.get("model_output", ""))

    points: list[dict[str, Any]] = []
    drags: list[dict[str, Any]] = []
    typed_inputs: list[dict[str, Any]] = []
    click_number = 0
    last_point: Optional[dict[str, Any]] = None
    for action_number, action in enumerate(actions, start=1):
        action_type = action.get("type")
        if action_type in {"Click", "LongPress"}:
            click_number += 1
            x, y = denormalize(
                action["x"],
                action["y"],
                width,
                height,
                coord_scale,
            )
            last_point = {
                "x": x,
                "y": y,
                "number": click_number,
                "kind": action_type,
                "text": None,
                "texts": [],
            }
            points.append(last_point)
        elif action_type == "Type":
            content = str(action.get("content", ""))
            typed_inputs.append(
                {
                    "number": action_number,
                    "content": content,
                    "targetNumber": (
                        last_point["number"] if last_point is not None else None
                    ),
                }
            )
            if last_point is not None:
                last_point["text"] = content
                last_point["texts"].append(content)
        elif action_type == "Drag":
            start_x, start_y = denormalize(
                action["sx"],
                action["sy"],
                width,
                height,
                coord_scale,
            )
            end_x, end_y = denormalize(
                action["ex"],
                action["ey"],
                width,
                height,
                coord_scale,
            )
            drags.append(
                {
                    "sx": start_x,
                    "sy": start_y,
                    "ex": end_x,
                    "ey": end_y,
                }
            )
            last_point = None

    return {
        "imagePath": os.path.relpath(image_path, relative_base),
        "imageSource": image_to_data_url(image_path),
        "width": width,
        "height": height,
        "coordScale": coord_scale,
        "task": result.get("task"),
        "modelOutput": result.get("model_output", ""),
        "reasoning": result.get("reasoning_content"),
        "inference": result.get("inference"),
        "actions": actions,
        "points": points,
        "drags": drags,
        "typedInputs": typed_inputs,
    }


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>单图 CAPTCHA 推理可视化</title>
<style>
:root { color-scheme:light; --bg:#f4f7fb; --card:#fff; --line:#dfe5ef; --text:#172033; --muted:#667085; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { width:min(1200px,calc(100% - 28px)); margin:22px auto; }
h1 { margin:0 0 14px; font-size:23px; }
.layout { display:grid; grid-template-columns:minmax(0,1.5fr) minmax(320px,.8fr); gap:16px; align-items:start; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; box-shadow:0 3px 12px rgba(16,24,40,.05); overflow:hidden; }
.card h2 { font-size:15px; margin:0; padding:11px 14px; border-bottom:1px solid var(--line); }
.canvas-wrap { padding:14px; display:flex; justify-content:center; overflow:auto; background:linear-gradient(135deg,#fafafa,#eef2f7); }
canvas { max-width:100%; height:auto; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,.18); }
.content { padding:12px 14px; }
.path { color:var(--muted); font-size:12px; overflow-wrap:anywhere; margin-bottom:8px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }
details { border-top:1px solid #edf0f5; padding-top:9px; margin-top:9px; }
summary { cursor:pointer; font-weight:650; }
pre { max-height:360px; overflow:auto; white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid #e5e9f0; border-radius:7px; padding:9px; font:12px/1.5 ui-monospace,SFMono-Regular,monospace; }
.actions { display:grid; gap:7px; }
.action { padding:8px 10px; border:1px solid #e4e8ef; border-radius:7px; background:#fafbfc; font-family:ui-monospace,SFMono-Regular,monospace; font-size:12px; }
.action.type-action { border-color:#a7f3d0; background:#ecfdf5; }
.typed-inputs { display:grid; gap:6px; margin-top:10px; }
.typed-input { padding:8px 10px; border-left:3px solid #10b981; border-radius:5px; background:#ecfdf5; color:#065f46; font-family:ui-monospace,SFMono-Regular,monospace; font-size:12px; white-space:pre-wrap; overflow-wrap:anywhere; }
.empty { color:#b42318; padding:8px 0; }
@media (max-width:820px) { .layout { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
  <h1>单图 CAPTCHA 推理可视化</h1>
  <div class="layout">
    <section class="card">
      <h2>图片与动作坐标</h2>
      <div class="canvas-wrap"><canvas id="canvas"></canvas></div>
      <div class="content">
        <div class="legend">
          <span><i class="dot" style="background:#f59e0b"></i>Click</span>
          <span><i class="dot" style="background:#0ea5e9"></i>LongPress</span>
          <span><i class="dot" style="background:#10b981"></i>Type</span>
          <span><i class="dot" style="background:#8b5cf6"></i>Drag</span>
        </div>
        <div class="path" id="path"></div>
      </div>
    </section>
    <aside class="card">
      <h2>推理与解析</h2>
      <div class="content">
        <div id="actions" class="actions"></div>
        <div id="typed-inputs" class="typed-inputs"></div>
        <details open><summary>模型输出</summary><pre id="model-output"></pre></details>
        <details id="reasoning-block"><summary>Reasoning</summary><pre id="reasoning"></pre></details>
        <details><summary>推理元数据</summary><pre id="metadata"></pre></details>
      </div>
    </aside>
  </div>
</main>
<script id="visual-data" type="application/json">__VISUAL_DATA__</script>
<script>
(function () {
  "use strict";
  var data=JSON.parse(document.getElementById("visual-data").textContent);
  document.getElementById("path").textContent=data.imagePath+" · "+data.width+"×"+data.height+" · coord-scale "+data.coordScale;
  document.getElementById("model-output").textContent=data.modelOutput || "";
  document.getElementById("reasoning").textContent=data.reasoning || "";
  document.getElementById("metadata").textContent=JSON.stringify(data.inference || {},null,2);
  if(!data.reasoning)document.getElementById("reasoning-block").style.display="none";
  var actions=document.getElementById("actions");
  if(!data.actions.length){var empty=document.createElement("div");empty.className="empty";empty.textContent="未解析出动作";actions.appendChild(empty);}
  data.actions.forEach(function(action,index){var row=document.createElement("div");row.className="action"+(action.type==="Type"?" type-action":"");row.textContent=(index+1)+". "+JSON.stringify(action);actions.appendChild(row);});
  var typedInputs=document.getElementById("typed-inputs");
  (data.typedInputs || []).forEach(function(item){
    var row=document.createElement("div");row.className="typed-input";
    var target=item.targetNumber===null?"无关联坐标":"关联点位 #"+item.targetNumber;
    row.textContent="Type #"+item.number+" ("+target+"): "+item.content;typedInputs.appendChild(row);
  });
  if(!(data.typedInputs || []).length)typedInputs.style.display="none";
  var canvas=document.getElementById("canvas");canvas.width=data.width;canvas.height=data.height;
  var ctx=canvas.getContext("2d");var image=new Image();
  image.onload=function(){
    ctx.drawImage(image,0,0,data.width,data.height);
    data.drags.forEach(function(d){
      var angle=Math.atan2(d.ey-d.sy,d.ex-d.sx);var head=Math.max(12,data.width/55);
      ctx.strokeStyle="#8b5cf6";ctx.fillStyle="#8b5cf6";ctx.lineWidth=Math.max(3,data.width/350);
      ctx.beginPath();ctx.moveTo(d.sx,d.sy);ctx.lineTo(d.ex,d.ey);ctx.stroke();
      ctx.beginPath();ctx.moveTo(d.ex,d.ey);
      ctx.lineTo(d.ex-head*Math.cos(angle-Math.PI/6),d.ey-head*Math.sin(angle-Math.PI/6));
      ctx.lineTo(d.ex-head*Math.cos(angle+Math.PI/6),d.ey-head*Math.sin(angle+Math.PI/6));
      ctx.closePath();ctx.fill();
    });
    data.points.forEach(function(point){
      var radius=Math.max(10,data.width/70);var color=point.kind==="LongPress"?"#0ea5e9":"#f59e0b";
      ctx.beginPath();ctx.arc(point.x,point.y,radius,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
      ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.stroke();
      ctx.fillStyle="#fff";ctx.textAlign="center";ctx.textBaseline="middle";
      ctx.font="bold "+Math.max(12,data.width/65)+"px sans-serif";ctx.fillText(String(point.number),point.x,point.y);
      var texts=Array.isArray(point.texts)?point.texts:(point.text==null?[]:[point.text]);
      texts.forEach(function(text,textIndex){
        var label="Type: "+text;ctx.font=Math.max(12,data.width/70)+"px sans-serif";
        var labelWidth=ctx.measureText(label).width+12;var labelX=Math.min(point.x+radius+5,data.width-labelWidth-3);
        var labelY=Math.max(3,point.y-radius-2+textIndex*27);ctx.fillStyle="rgba(5,150,105,.92)";
        ctx.fillRect(labelX,labelY,labelWidth,24);ctx.fillStyle="#fff";ctx.textAlign="left";ctx.textBaseline="middle";
        ctx.fillText(label,labelX+6,labelY+12);
      });
    });
  };
  image.src=data.imageSource;
}());
</script>
</body>
</html>
"""


def generate_html(payload: dict[str, Any], output_path: str) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).replace(
        "<",
        "\\u003c",
    ).replace("&", "\\u0026")
    rendered = _HTML_TEMPLATE.replace("__VISUAL_DATA__", encoded)
    output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as destination:
        destination.write(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 infer_captcha.py 的单图 JSON 结果可视化为 HTML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result", required=True, help="单图推理结果 JSON")
    parser.add_argument("--image", default=None, help="覆盖结果中的原图路径")
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=None,
        help="覆盖结果中的坐标上界",
    )
    parser.add_argument(
        "--output",
        default="test_visualization.html",
        help="输出 HTML 文件",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result_path = os.path.abspath(os.path.expanduser(args.result))
    output_path = os.path.abspath(os.path.expanduser(args.output))
    if not os.path.isfile(result_path):
        parser.error("--result 不存在或不是文件: %s" % args.result)
    try:
        result = load_result(result_path)
        payload = build_visual_payload(
            result=result,
            result_path=result_path,
            image_override=args.image,
            coord_scale_override=args.coord_scale,
            path_base=os.path.dirname(output_path),
        )
        generate_html(payload, output_path)
        print(os.path.relpath(output_path))
        return 0
    except Exception as error:
        print("可视化失败: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
