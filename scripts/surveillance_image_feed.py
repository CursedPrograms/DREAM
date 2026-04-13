#!/usr/bin/env python3
"""
surveillance_image_feed.py
──────────────────────────
Runs the full AI surveillance analysis pipeline on still images.

Modes:
  Single image    python3 surveillance_image_feed.py photo.jpg
  Multiple images python3 surveillance_image_feed.py img1.jpg img2.jpg img3.jpg
  Whole folder    python3 surveillance_image_feed.py --folder /path/to/images/
  Watch folder    python3 surveillance_image_feed.py --watch /path/to/folder/
                  (stays running, processes new images as they arrive)

For each image it produces:
  output/image_feed/<name>_annotated.jpg   — annotated image with boxes + labels
  output/image_feed/json/<name>.json       — full structured report

Options:
  --show          Display each annotated image in a window (press any key to advance)
  --conf 0.4      Detection confidence threshold (default: 0.45)
  --output DIR    Custom output directory
  --extensions    Comma-separated image extensions to watch (default: jpg,jpeg,png,bmp,webp)
"""

import argparse
import os
import sys
import time
import glob
import cv2
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box as rbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surveillance_core import analyse_frame, save_json, Models, console

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Smart Image Surveillance Analyser")
    p.add_argument("images",      nargs="*",                   help="Image file(s) to analyse")
    p.add_argument("--folder",    type=str,  default=None,     help="Analyse all images in a folder")
    p.add_argument("--watch",     type=str,  default=None,     help="Watch folder for new images continuously")
    p.add_argument("--show",      action="store_true",         help="Display annotated result in a window")
    p.add_argument("--conf",      type=float, default=0.45,    help="Confidence threshold (default: 0.45)")
    p.add_argument("--output",    type=str,  default=None,     help="Output directory")
    p.add_argument("--extensions",type=str,  default="jpg,jpeg,png,bmp,webp",
                                                                help="Extensions to scan (comma-separated)")
    p.add_argument("--no-save",   action="store_true",         help="Do not save output files, just print results")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def collect_images_from_folder(folder, extensions):
    paths = []
    for ext in extensions:
        paths += glob.glob(os.path.join(folder, f"*.{ext}"))
        paths += glob.glob(os.path.join(folder, f"*.{ext.upper()}"))
    return sorted(set(paths))

def print_report(report, image_path):
    """Pretty-print analysis report to console."""
    console.print()
    console.print(f"[bold white]{'─'*60}[/bold white]")
    console.print(f"[bold cyan]Image:[/bold cyan] {os.path.basename(image_path)}")
    console.print(f"[bold cyan]Scene:[/bold cyan] {report['scene_summary']}")
    console.print(f"[bold cyan]Threat:[/bold cyan] {report['threat_level']}")

    if report["people"]:
        table = Table(title="People Detected", box=rbox.SIMPLE, show_lines=True)
        table.add_column("#",         style="dim",    width=3)
        table.add_column("Conf",      style="yellow", width=6)
        table.add_column("Age",       style="green",  width=10)
        table.add_column("Gender",    style="cyan",   width=18)
        table.add_column("Emotion",   style="magenta",width=20)
        table.add_column("BBox (x1,y1→x2,y2)", style="dim", width=22)

        for i, p in enumerate(report["people"], 1):
            b = p["bbox"]
            gender_str  = f"{p['gender']} ({p['gender_conf']*100:.0f}%)" if p['gender'] != 'unknown' else "unknown"
            emotion_str = f"{p['emotion']} ({p['emotion_conf']*100:.0f}%)" if p['emotion'] != 'unknown' else "unknown"
            table.add_row(
                str(i),
                f"{p['confidence']*100:.0f}%",
                p["age"],
                gender_str,
                emotion_str,
                f"({b['x1']},{b['y1']})→({b['x2']},{b['y2']})",
            )
        console.print(table)

    if report["animals"]:
        table = Table(title="Animals Detected", box=rbox.SIMPLE)
        table.add_column("#",       style="dim",    width=3)
        table.add_column("Species", style="cyan",   width=20)
        table.add_column("Conf",    style="yellow", width=6)
        table.add_column("BBox",    style="dim",    width=22)
        for i, a in enumerate(report["animals"], 1):
            b = a["bbox"]
            table.add_row(str(i), a["species"], f"{a['confidence']*100:.0f}%",
                          f"({b['x1']},{b['y1']})→({b['x2']},{b['y2']})")
        console.print(table)

    if report["vehicles"]:
        table = Table(title="Vehicles Detected", box=rbox.SIMPLE)
        table.add_column("#",    style="dim",    width=3)
        table.add_column("Type", style="magenta",width=20)
        table.add_column("Conf", style="yellow", width=6)
        table.add_column("BBox", style="dim",    width=22)
        for i, v in enumerate(report["vehicles"], 1):
            b = v["bbox"]
            table.add_row(str(i), v["type"], f"{v['confidence']*100:.0f}%",
                          f"({b['x1']},{b['y1']})→({b['x2']},{b['y2']})")
        console.print(table)

    if not report["people"] and not report["animals"] and not report["vehicles"]:
        console.print("[dim]  No persons, animals, or vehicles detected.[/dim]")

# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path, out_dir, json_dir, args):
    """
    Loads, analyses, annotates, and saves one image.
    Returns the report dict, or None on failure.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        console.print(f"[red]Cannot read image: {image_path}[/red]")
        return None

    source_name = os.path.basename(image_path)
    annotated, report = analyse_frame(frame, source_name)

    # Save outputs
    if not args.no_save:
        stem = os.path.splitext(source_name)[0]
        ann_path  = os.path.join(out_dir,  f"{stem}_annotated.jpg")
        json_path = os.path.join(json_dir, f"{stem}.json")

        cv2.imwrite(ann_path, annotated)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        console.print(f"[green]  ✔ Annotated:[/green] {ann_path}")
        console.print(f"[green]  ✔ Report   :[/green] {json_path}")

    print_report(report, image_path)

    # Optionally display
    if args.show:
        win = f"Surveillance — {source_name}  [any key = next  |  Q = quit]"
        cv2.imshow(win, annotated)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q') or key == 27:
            return None   # signal to stop

    return report

# ─────────────────────────────────────────────────────────────────────────────
#  WATCH MODE
# ─────────────────────────────────────────────────────────────────────────────

def watch_folder(folder, out_dir, json_dir, args, extensions):
    """
    Continuously monitors a folder and processes any new image files.
    Keeps a set of already-processed files so it never re-processes.
    """
    console.print(f"[bold cyan]Watching:[/bold cyan] {folder}")
    console.print("[dim]Drop images into the folder — they will be analysed automatically.[/dim]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    console.print()

    processed = set(collect_images_from_folder(folder, extensions))
    console.print(f"[dim]{len(processed)} existing file(s) skipped.[/dim]")

    try:
        while True:
            current = set(collect_images_from_folder(folder, extensions))
            new_files = current - processed
            for path in sorted(new_files):
                console.print(f"\n[bold yellow]New image:[/bold yellow] {os.path.basename(path)}")
                # Wait briefly in case the file is still being written
                time.sleep(0.5)
                process_image(path, out_dir, json_dir, args)
                processed.add(path)
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Watch stopped.[/yellow]")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    extensions = [e.strip().lstrip(".") for e in args.extensions.split(",")]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR  = args.output or os.path.join(BASE_DIR, "output", "image_feed")
    JSON_DIR = os.path.join(OUT_DIR, "json")

    if not args.no_save:
        os.makedirs(OUT_DIR,  exist_ok=True)
        os.makedirs(JSON_DIR, exist_ok=True)

    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║  Smart Surveillance — Image Feed     ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════╝[/bold cyan]")
    console.print()

    # Pre-load all models
    Models.get()
    console.print()

    # ── Collect image paths ───────────────────────────────────────────────────
    if args.watch:
        if not os.path.isdir(args.watch):
            console.print(f"[red]Watch folder does not exist: {args.watch}[/red]")
            sys.exit(1)
        watch_folder(args.watch, OUT_DIR, JSON_DIR, args, extensions)
        return

    image_paths = list(args.images)

    if args.folder:
        if not os.path.isdir(args.folder):
            console.print(f"[red]Folder does not exist: {args.folder}[/red]")
            sys.exit(1)
        found = collect_images_from_folder(args.folder, extensions)
        console.print(f"[cyan]Found {len(found)} image(s) in {args.folder}[/cyan]")
        image_paths += found

    if not image_paths:
        console.print("[red]No images specified. Use:[/red]")
        console.print("  python3 surveillance_image_feed.py image.jpg")
        console.print("  python3 surveillance_image_feed.py --folder /path/to/images/")
        console.print("  python3 surveillance_image_feed.py --watch /path/to/folder/")
        sys.exit(1)

    # ── Process ───────────────────────────────────────────────────────────────
    total   = len(image_paths)
    success = 0
    all_reports = []

    console.print(f"[bold]Processing {total} image(s)...[/bold]")
    console.print()

    for i, path in enumerate(image_paths, 1):
        if not os.path.isfile(path):
            console.print(f"[red]File not found: {path}[/red]")
            continue
        console.print(f"[bold white][{i}/{total}][/bold white] {os.path.basename(path)}")
        report = process_image(path, OUT_DIR, JSON_DIR, args)
        if report is None:
            # User pressed Q in show mode
            console.print("[yellow]Stopped by user.[/yellow]")
            break
        all_reports.append(report)
        success += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.print(f"[bold cyan]{'─'*60}[/bold cyan]")
    console.print(f"[bold]Batch complete:[/bold] {success}/{total} images processed")

    total_people  = sum(len(r["people"])   for r in all_reports)
    total_animals = sum(len(r["animals"])  for r in all_reports)
    total_vehicles= sum(len(r["vehicles"]) for r in all_reports)

    console.print(f"  People detected  : {total_people}")
    console.print(f"  Animals detected : {total_animals}")
    console.print(f"  Vehicles detected: {total_vehicles}")

    if not args.no_save and all_reports:
        # Save a combined batch report
        batch_path = os.path.join(JSON_DIR, f"batch_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(batch_path, "w") as f:
            json.dump({
                "batch_timestamp": datetime.now().isoformat(),
                "total_images":    total,
                "processed":       success,
                "total_people":    total_people,
                "total_animals":   total_animals,
                "total_vehicles":  total_vehicles,
                "reports":         all_reports,
            }, f, indent=2)
        console.print(f"\n[green]Batch report saved:[/green] {batch_path}")

    console.print()


if __name__ == "__main__":
    main()
