import json
import os
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

def is_correct(pred, bbox):
    """Check if the predicted point is inside the ground truth bbox."""
    if not pred or not bbox:
        return False
    x, y = pred
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def resolve_image_path(json_img_path, image_root=None):
    """
    Try to resolve the image path. 
    Handles cases where the JSON contains server-side absolute paths.
    """
    if os.path.exists(json_img_path):
        return json_img_path
    
    # Try to find relative to common patterns
    for pattern in ["Screenspot-pro/images/", "images/"]:
        if pattern in json_img_path:
            rel_path = json_img_path.split(pattern)[-1]
            # Try current workspace locations if image_root not provided
            search_paths = [image_root] if image_root else [
                ".", 
                "../", 
                "/Users/syoya/Documents/UI-Venus/assets" # Example potential location
            ]
            for root in search_paths:
                if not root: continue
                potential_path = os.path.join(root, rel_path)
                if os.path.exists(potential_path):
                    return potential_path
    return None

def draw_on_image(img, pred, bbox, is_correct_flag, title, zoom_in_info=None):
    """Draw annotations on a single image frame."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # Draw GT Box (Green)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="#28a745", width=max(3, int(h/200)))
    
    # Draw Zoom-in Crop Box (Blue dashed-like)
    if zoom_in_info and 'crop_box' in zoom_in_info:
        cx1, cy1, cx2, cy2 = zoom_in_info['crop_box']
        # Simple dashed effect
        draw.rectangle([cx1, cy1, cx2, cy2], outline="#007bff", width=max(2, int(h/300)))

    # Draw Prediction (Red/Green Dot)
    px, py = pred
    r = max(5, int(h/100))
    color = "#28a745" if is_correct_flag else "#dc3545"
    draw.ellipse([px-r, py-r, px+r, py+r], fill=color, outline="white", width=2)
    
    # Draw Label Text
    try:
        # Try to use a larger font if available, else default
        font = ImageFont.load_default()
    except:
        font = None
    
    label = f"{title}: {'PASS' if is_correct_flag else 'FAIL'}"
    draw.text((20, 20), label, fill=color, font=font, stroke_width=2, stroke_fill="black")
    return img

def create_side_by_side(item1, item2, output_path, image_root=None):
    """Generate a side-by-side comparison PNG."""
    raw_path = resolve_image_path(item1['img_path'], image_root)
    
    if not raw_path:
        # Create a placeholder if image not found
        img_raw = Image.new('RGB', (1200, 800), color='#2c3e50')
        d = ImageDraw.Draw(img_raw)
        d.text((100, 400), f"IMAGE NOT FOUND: {os.path.basename(item1['img_path'])}", fill="white")
    else:
        img_raw = Image.open(raw_path).convert('RGB')
    
    # Create two copies
    img_left = img_raw.copy()
    img_right = img_raw.copy()
    
    # Annotate Left (Zoom-In)
    draw_on_image(img_left, item1['pred'], item1['org_info']['bbox'], 
                 item1['is_correct'], "Zoom-In (0.5)", item1.get('zoom_in_info'))
    
    # Annotate Right (Baseline)
    draw_on_image(img_right, item2['pred'], item2['org_info']['bbox'], 
                 item2['is_correct'], "Baseline")
    
    # Combine
    w, h = img_left.size
    combined = Image.new('RGB', (w * 2 + 20, h), "#ecf0f1")
    combined.paste(img_left, (0, 0))
    combined.paste(img_right, (w + 20, 0))
    
    # Resize for reasonably sized output if huge
    if w > 1600:
        scale = 1600 / w
        combined = combined.resize((int(combined.width * scale), int(combined.height * scale)), Image.Resampling.LANCZOS)
        
    combined.save(output_path)

def generate_html_report(flips, output_dir, total_samples, stats):
    """Generate HTML referencing the generated PNGs."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ScreenSpot-Pro Visual Comparison</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #f5f7f9; }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 40px 20px; }}
            header {{ background: #2c3e50; color: white; padding: 30px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stats-bar {{ display: flex; gap: 20px; margin-top: 20px; }}
            .stat-box {{ background: rgba(255,255,255,0.1); padding: 10px 20px; border-radius: 6px; }}
            .sample-card {{ background: white; border-radius: 12px; margin-bottom: 50px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); overflow: hidden; border: 1px solid #e1e4e8; }}
            .card-header {{ padding: 15px 25px; background: #f8f9fa; border-bottom: 1px solid #e1e4e8; display: flex; justify-content: space-between; align-items: center; }}
            .card-body {{ padding: 25px; }}
            .instruction {{ font-size: 1.2em; font-weight: bold; margin-bottom: 15px; color: #2c3e50; }}
            .meta-info {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
            .comparison-img {{ width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
            .tag {{ padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8em; text-transform: uppercase; }}
            .improved {{ background: #2ecc71; color: white; }}
            .degraded {{ background: #e74c3c; color: white; }}
        </style>
    </head>
    <body>
        <header>
            <div class="container">
                <h1>ScreenSpot-Pro: Zoom-In vs Baseline</h1>
                <div class="stats-bar">
                    <div class="stat-box">Total Samples: {total_samples}</div>
                    <div class="stat-box" style="border-left: 4px solid #2ecc71;">Improved: {stats['improved']}</div>
                    <div class="stat-box" style="border-left: 4px solid #e74c3c;">Degraded: {stats['degraded']}</div>
                </div>
            </div>
        </header>
        <div class="container">
    """
    
    for flip in flips:
        tag_class = "improved" if flip['type'] == 'improved' else "degraded"
        html_content += f"""
            <div class="sample-card">
                <div class="card-header">
                    <span>ID: {flip['id']}</span>
                    <span class="tag {tag_class}">{flip['type']}</span>
                </div>
                <div class="card-body">
                    <div class="instruction">"{flip['instruction']}"</div>
                    <div class="meta-info">
                        Application: <strong>{flip['application']}</strong> | 
                        Platform: <strong>{flip['platform']}</strong> | 
                        UI Type: <strong>{flip['ui_type']}</strong>
                    </div>
                    <img class="comparison-img" src="viz/{flip['id']}.png" alt="Comparison">
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoomin_json", default="merge_68_72_72_ties_weighted_zoomin05.json")
    parser.add_argument("--baseline_json", default="merge_68_72_72_ties_weighted.json")
    parser.add_argument("--output_dir", default="compare_results_pil")
    parser.add_argument("--image_root", help="Root folder where images are stored locally")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PNGs to generate (default: all)")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    viz_dir = os.path.join(args.output_dir, "viz")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    print(f"Loading data...")
    with open(args.zoomin_json) as f: d1 = json.load(f)
    with open(args.baseline_json) as f: d2 = json.load(f)

    map1 = {i['org_info']['id']: i for i in d1['details']}
    map2 = {i['org_info']['id']: i for i in d2['details']}
    common_ids = sorted(list(set(map1.keys()) & set(map2.keys())))

    flips = []
    stats = {"improved": 0, "degraded": 0, "both_correct": 0, "both_wrong": 0}
    
    for cid in common_ids:
        i1, i2 = map1[cid], map2[cid]
        i1['is_correct'] = is_correct(i1['pred'], i1['org_info']['bbox'])
        i2['is_correct'] = is_correct(i2['pred'], i2['org_info']['bbox'])
        
        ftype = None
        if i1['is_correct'] and not i2['is_correct']:
            stats['improved'] += 1
            ftype = "improved"
        elif not i1['is_correct'] and i2['is_correct']:
            stats['degraded'] += 1
            ftype = "degraded"
        elif i1['is_correct'] and i2['is_correct']:
            stats['both_correct'] += 1
        else:
            stats['both_wrong'] += 1
            
        if ftype:
            flips.append({
                "id": cid, "type": ftype, "instruction": i1['org_info']['instruction'],
                "application": i1['org_info']['application'], "platform": i1['org_info']['platform'],
                "ui_type": i1['org_info']['ui_type'], "img_path": i1['img_path'],
                "item1": i1, "item2": i2
            })

    print(f"Stats: Improved={stats['improved']}, Degraded={stats['degraded']}")
    
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Generate images
    limit = args.limit if args.limit is not None else len(flips)
    to_process = flips[:limit]
    print(f"Generating {len(to_process)} comparison images...")
    for i, flip in enumerate(to_process):
        out_path = os.path.join(viz_dir, f"{flip['id']}.png")
        create_side_by_side(flip['item1'], flip['item2'], out_path, args.image_root)
        if (i+1) % 10 == 0: print(f"  Processed {i+1}/{len(to_process)}")

    generate_html_report(to_process, args.output_dir, len(common_ids), stats)
    print(f"\nCompleted! Open '{args.output_dir}/report.html' to view results.")

if __name__ == "__main__":
    main()
