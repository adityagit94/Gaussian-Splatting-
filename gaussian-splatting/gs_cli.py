#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List


# ----------------------------
# Helpers
# ----------------------------
def run_cmd(cmd: List[str], log_path: Path, cwd: Optional[Path] = None):
    cmd_str = " ".join(map(str, cmd))
    print(f"\n[CMD] {cmd_str}\n")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(cmd_str + "\n")
        f.write("=" * 120 + "\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed (exit={rc}): {cmd_str}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_paths(workspace: Path):
    return {
        "workspace": workspace,
        "images": workspace / "images",
        "colmap": workspace / "colmap",
        "db": workspace / "colmap" / "database.db",
        "sparse": workspace / "colmap" / "sparse",
        "sparse0": workspace / "colmap" / "sparse" / "0",
        "undistorted": workspace / "undistorted",
        "gs_out": workspace / "gs_out",
        "runs": workspace / "runs",
    }


def wsl_to_windows_path(p: Path) -> str:
    r"""Convert /mnt/c/... to C:\..."""
    p_str = str(p)
    if p_str.startswith("/mnt/") and len(p_str) > 6:
        drive = p_str[5].upper()
        rest = p_str[6:]
        rest = rest.replace("/", "\\")
        return f"{drive}:{rest}"
    return p_str


def fix_sparse0_bins(undistorted: Path):
    """
    Some setups end up with .bin sitting in undistorted/sparse/ instead of undistorted/sparse/0.
    Move cameras.bin/images.bin/points3D.bin into sparse/0 if needed.
    """
    sparse = undistorted / "sparse"
    sparse0 = sparse / "0"
    ensure_dir(sparse0)
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = sparse / name
        dst = sparse0 / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))


def colmap_has_option(cmd_name: str, option: str) -> bool:
    """
    True if `colmap <cmd_name> -h` output contains the option string.
    """
    try:
        r = subprocess.run(
            ["colmap", cmd_name, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return option in (r.stdout or "")
    except Exception:
        return False


def add_feature_extraction_gpu_flags(cmd: List[str], use_gpu: bool) -> List[str]:
    """
    COLMAP builds differ:
      - Newer: FeatureExtraction.use_gpu / FeatureExtraction.gpu_index
      - Older: SiftExtraction.use_gpu / SiftExtraction.gpu_index
    Prefer newer if present.
    """
    if colmap_has_option("feature_extractor", "--FeatureExtraction.use_gpu"):
        cmd += ["--FeatureExtraction.use_gpu", "1" if use_gpu else "0"]
        if colmap_has_option("feature_extractor", "--FeatureExtraction.gpu_index"):
            cmd += ["--FeatureExtraction.gpu_index", "0" if use_gpu else "-1"]
        return cmd

    if colmap_has_option("feature_extractor", "--SiftExtraction.use_gpu"):
        cmd += ["--SiftExtraction.use_gpu", "1" if use_gpu else "0"]
        if colmap_has_option("feature_extractor", "--SiftExtraction.gpu_index"):
            cmd += ["--SiftExtraction.gpu_index", "0" if use_gpu else "-1"]
        return cmd

    # fallback: gpu_index only
    if colmap_has_option("feature_extractor", "--FeatureExtraction.gpu_index"):
        cmd += ["--FeatureExtraction.gpu_index", "0" if use_gpu else "-1"]
        return cmd
    if colmap_has_option("feature_extractor", "--SiftExtraction.gpu_index"):
        cmd += ["--SiftExtraction.gpu_index", "0" if use_gpu else "-1"]
        return cmd

    print("[WARN] Could not find a known GPU flag for colmap feature_extractor; using defaults.")
    return cmd


def add_feature_matching_gpu_flags(matcher_cmd: str, cmd: List[str], use_gpu: bool) -> List[str]:
    """
    Newer: FeatureMatching.use_gpu / FeatureMatching.gpu_index
    Older: SiftMatching.use_gpu / SiftMatching.gpu_index
    """
    if colmap_has_option(matcher_cmd, "--FeatureMatching.use_gpu"):
        cmd += ["--FeatureMatching.use_gpu", "1" if use_gpu else "0"]
        if colmap_has_option(matcher_cmd, "--FeatureMatching.gpu_index"):
            cmd += ["--FeatureMatching.gpu_index", "0" if use_gpu else "-1"]
        return cmd

    if colmap_has_option(matcher_cmd, "--SiftMatching.use_gpu"):
        cmd += ["--SiftMatching.use_gpu", "1" if use_gpu else "0"]
        if colmap_has_option(matcher_cmd, "--SiftMatching.gpu_index"):
            cmd += ["--SiftMatching.gpu_index", "0" if use_gpu else "-1"]
        return cmd

    # fallback: gpu_index only
    if colmap_has_option(matcher_cmd, "--FeatureMatching.gpu_index"):
        cmd += ["--FeatureMatching.gpu_index", "0" if use_gpu else "-1"]
        return cmd
    if colmap_has_option(matcher_cmd, "--SiftMatching.gpu_index"):
        cmd += ["--SiftMatching.gpu_index", "0" if use_gpu else "-1"]
        return cmd

    print(f"[WARN] Could not find a known GPU flag for colmap {matcher_cmd}; using defaults.")
    return cmd


def add_mapper_ba_gpu_flags(cmd: List[str], mapper_ba_use_gpu: bool) -> List[str]:
    if colmap_has_option("mapper", "--Mapper.ba_use_gpu"):
        cmd += ["--Mapper.ba_use_gpu", "1" if mapper_ba_use_gpu else "0"]
    return cmd


def find_latest_iteration(gs_out: Path) -> Optional[int]:
    """
    Find latest saved gaussian iteration:
      gs_out/point_cloud/iteration_30000/...
    """
    pc_dir = gs_out / "point_cloud"
    if not pc_dir.exists():
        return None

    iters: List[int] = []
    for d in pc_dir.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                iters.append(int(d.name.split("_", 1)[1]))
            except Exception:
                pass
    return max(iters) if iters else None


def find_latest_pth_checkpoint(gs_out: Path) -> Optional[Path]:
    """
    Find latest .pth checkpoint anywhere inside gs_out (depth-limited).
    This is what train.py --start_checkpoint expects.
    """
    # common: gs_out/*.pth
    cands = list(gs_out.glob("*.pth"))

    # also check gs_out/point_cloud/iteration_*/ for some forks
    pc_dir = gs_out / "point_cloud"
    if pc_dir.exists():
        for d in pc_dir.iterdir():
            if d.is_dir() and d.name.startswith("iteration_"):
                cands += list(d.glob("*.pth"))

    if not cands:
        return None

    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


# ----------------------------
# Pipeline steps
# ----------------------------
def step_init(paths: dict):
    for k in ["images", "colmap", "sparse", "undistorted", "gs_out", "runs"]:
        ensure_dir(paths[k])


def step_extract(
    video: Path,
    paths: dict,
    fps: float,
    jpg_quality: int,
    skip_seconds: Optional[float],
    log_path: Path,
):
    ensure_dir(paths["images"])
    out_pat = str(paths["images"] / "frame_%05d.jpg")

    cmd = ["ffmpeg", "-y"]
    if skip_seconds is not None and skip_seconds > 0:
        cmd += ["-ss", str(skip_seconds)]
    cmd += [
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(jpg_quality),
        out_pat,
    ]
    run_cmd(cmd, log_path)


def step_colmap(
    paths: dict,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    matcher: str,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Optional[Path],
    mapper_ba_use_gpu: bool,
    log_path: Path,
):
    ensure_dir(paths["colmap"])
    ensure_dir(paths["sparse0"])

    # feature_extractor
    cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(paths["db"]),
        "--image_path",
        str(paths["images"]),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
    ]
    cmd = add_feature_extraction_gpu_flags(cmd, use_gpu)
    run_cmd(cmd, log_path)

    # matcher
    if matcher == "sequential":
        cmd = [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(paths["db"]),
            "--SequentialMatching.overlap",
            str(overlap),
        ]
        if loop_detection:
            cmd += ["--SequentialMatching.loop_detection", "1"]

        cmd = add_feature_matching_gpu_flags("sequential_matcher", cmd, use_gpu)
        run_cmd(cmd, log_path)

    elif matcher == "exhaustive":
        cmd = ["colmap", "exhaustive_matcher", "--database_path", str(paths["db"])]
        cmd = add_feature_matching_gpu_flags("exhaustive_matcher", cmd, use_gpu)
        run_cmd(cmd, log_path)

    elif matcher == "vocab":
        if vocab_tree_path is None:
            raise ValueError("matcher=vocab requires --vocab-tree /path/to/vocab_tree.bin")

        if colmap_has_option("vocab_tree_matcher", "--database_path"):
            cmd = [
                "colmap",
                "vocab_tree_matcher",
                "--database_path",
                str(paths["db"]),
                "--VocabTreeMatching.vocab_tree_path",
                str(vocab_tree_path),
            ]
            cmd = add_feature_matching_gpu_flags("vocab_tree_matcher", cmd, use_gpu)
            run_cmd(cmd, log_path)
        else:
            cmd = [
                "colmap",
                "sequential_matcher",
                "--database_path",
                str(paths["db"]),
                "--SequentialMatching.overlap",
                str(overlap),
                "--SequentialMatching.loop_detection",
                "1",
                "--SequentialMatching.vocab_tree_path",
                str(vocab_tree_path),
            ]
            cmd = add_feature_matching_gpu_flags("sequential_matcher", cmd, use_gpu)
            run_cmd(cmd, log_path)
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    # mapper
    cmd = [
        "colmap",
        "mapper",
        "--database_path",
        str(paths["db"]),
        "--image_path",
        str(paths["images"]),
        "--output_path",
        str(paths["sparse"]),
    ]
    cmd = add_mapper_ba_gpu_flags(cmd, mapper_ba_use_gpu)
    run_cmd(cmd, log_path)


def step_undistort(paths: dict, log_path: Path):
    ensure_dir(paths["undistorted"])
    cmd = [
        "colmap",
        "image_undistorter",
        "--image_path",
        str(paths["images"]),
        "--input_path",
        str(paths["sparse0"]),
        "--output_path",
        str(paths["undistorted"]),
        "--output_type",
        "COLMAP",
    ]
    run_cmd(cmd, log_path)
    fix_sparse0_bins(paths["undistorted"])


def step_train(
    gs_repo: Path,
    paths: dict,
    iterations: int,
    save_iters: List[int],
    checkpoint_iters: List[int],
    extra_args: List[str],
    log_path: Path,
    resume: bool = False,
):
    train_py = gs_repo / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found at: {train_py}")

    ensure_dir(paths["gs_out"])

    cmd = [
        "python",
        str(train_py),
        "-s",
        str(paths["undistorted"]),
        "-m",
        str(paths["gs_out"]),
        "--iterations",
        str(iterations),
    ]

    if save_iters:
        cmd += ["--save_iterations"] + [str(x) for x in save_iters]

    if checkpoint_iters:
        cmd += ["--checkpoint_iterations"] + [str(x) for x in checkpoint_iters]

    if resume:
        ckpt = find_latest_pth_checkpoint(paths["gs_out"])
        if ckpt is None:
            latest_gauss = find_latest_iteration(paths["gs_out"])
            raise RuntimeError(
                "No .pth checkpoint found to resume from.\n"
                "You currently only have saved gaussians (PLY) like:\n"
                f"  gs_out/point_cloud/iteration_{latest_gauss}/point_cloud.ply\n"
                "To enable resume, re-train with --checkpoint-iters (or pass via --extra --checkpoint_iterations ...)."
            )
        print(f"[INFO] Resuming from checkpoint: {ckpt}")
        cmd += ["--start_checkpoint", str(ckpt)]

    cmd += extra_args
    run_cmd(cmd, log_path, cwd=gs_repo)


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(
        description="Local Gaussian Splatting CLI pipeline (ffmpeg + COLMAP + train.py)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(a):
        a.add_argument("--workspace", type=Path, required=True, help="Workspace folder for this object/scene")

    # init
    a = sub.add_parser("init")
    add_common(a)

    # extract
    a = sub.add_parser("extract")
    add_common(a)
    a.add_argument("--video", type=Path, required=True)
    a.add_argument("--fps", type=float, default=3.0)
    a.add_argument("--jpg-quality", type=int, default=2, help="1 best, 31 worst")
    a.add_argument("--skip-seconds", type=float, default=0.0)

    # colmap
    a = sub.add_parser("colmap")
    add_common(a)
    a.add_argument("--camera-model", type=str, default="SIMPLE_RADIAL")
    a.add_argument("--single-camera", action="store_true", default=True)
    a.add_argument("--no-single-camera", dest="single_camera", action="store_false")
    a.add_argument("--use-gpu", action="store_true", default=True)
    a.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    a.add_argument("--matcher", choices=["sequential", "exhaustive", "vocab"], default="sequential")
    a.add_argument("--overlap", type=int, default=20)
    a.add_argument("--loop-detection", action="store_true", default=False)
    a.add_argument("--vocab-tree", type=Path, default=None)
    a.add_argument("--mapper-ba-use-gpu", action="store_true", default=True)
    a.add_argument("--no-mapper-ba-gpu", dest="mapper_ba_use_gpu", action="store_false")

    # undistort
    a = sub.add_parser("undistort")
    add_common(a)

    # train
    a = sub.add_parser("train")
    add_common(a)
    a.add_argument("--gs-repo", type=Path, required=True, help="Path to gaussian-splatting repo (contains train.py)")
    a.add_argument("--iterations", type=int, default=30000)
    a.add_argument("--save-iters", type=int, nargs="*", default=[7000, 10000, 15000, 20000, 30000])
    a.add_argument("--checkpoint-iters", type=int, nargs="*", default=[], help="Save .pth checkpoints at these iterations")
    a.add_argument("--resume", action="store_true", default=False, help="Resume from latest .pth checkpoint in gs_out/")
    a.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra args passed to train.py after '--extra'")

    # run (all-in-one)
    a = sub.add_parser("run")
    add_common(a)
    a.add_argument("--video", type=Path, required=False, help="If provided, frames extracted into workspace/images")
    a.add_argument("--images-dir", type=Path, required=False, help="If provided, copied into workspace/images (if empty)")
    a.add_argument("--fps", type=float, default=3.0)
    a.add_argument("--jpg-quality", type=int, default=2)
    a.add_argument("--skip-seconds", type=float, default=0.0)

    a.add_argument("--camera-model", type=str, default="SIMPLE_RADIAL")
    a.add_argument("--single-camera", action="store_true", default=True)
    a.add_argument("--no-single-camera", dest="single_camera", action="store_false")
    a.add_argument("--use-gpu", action="store_true", default=True)
    a.add_argument("--no-gpu", dest="use_gpu", action="store_false")

    a.add_argument("--matcher", choices=["sequential", "exhaustive", "vocab"], default="sequential")
    a.add_argument("--overlap", type=int, default=20)
    a.add_argument("--loop-detection", action="store_true", default=False)
    a.add_argument("--vocab-tree", type=Path, default=None)
    a.add_argument("--mapper-ba-use-gpu", action="store_true", default=True)
    a.add_argument("--no-mapper-ba-gpu", dest="mapper_ba_use_gpu", action="store_false")

    a.add_argument("--gs-repo", type=Path, required=True)
    a.add_argument("--iterations", type=int, default=30000)
    a.add_argument("--save-iters", type=int, nargs="*", default=[7000, 10000, 15000, 20000, 30000])
    a.add_argument("--checkpoint-iters", type=int, nargs="*", default=[], help="Save .pth checkpoints at these iterations")
    a.add_argument("--resume", action="store_true", default=False, help="Resume from latest .pth checkpoint in gs_out/")
    a.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    # viewcmd
    a = sub.add_parser("viewcmd")
    add_common(a)
    a.add_argument("--iter", type=int, required=True, help="Iteration to view (e.g. 30000)")
    a.add_argument(
        "--viewer-dir",
        type=Path,
        default=Path("/mnt/c/Users/Guest_/OneDrive/Desktop/viewers/bin"),
        help="WSL path to the Windows SIBR viewer directory (/mnt/c/...)",
    )

    args = p.parse_args()
    paths = default_paths(args.workspace)
    step_init(paths)

    run_id = timestamp_id()
    log_path = paths["runs"] / f"{run_id}_log.txt"
    meta_path = paths["runs"] / f"{run_id}_run.json"
    meta = {"run_id": run_id, "args": vars(args)}
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    if args.cmd == "init":
        print(f"Initialized workspace: {paths['workspace']}")
        return

    if args.cmd == "viewcmd":
        viewer_exe = args.viewer_dir / "SIBR_gaussianViewer_app.exe"
        win_exe = wsl_to_windows_path(viewer_exe)
        win_model = wsl_to_windows_path(paths["gs_out"])
        print("\nCopy/paste into Windows CMD or PowerShell:\n")
        print(f'{win_exe} --model-path "{win_model}" --iteration {args.iter}')
        return

    if args.cmd == "extract":
        step_extract(args.video, paths, args.fps, args.jpg_quality, args.skip_seconds, log_path)
        return

    if args.cmd == "colmap":
        step_colmap(
            paths=paths,
            camera_model=args.camera_model,
            single_camera=args.single_camera,
            use_gpu=args.use_gpu,
            matcher=args.matcher,
            overlap=args.overlap,
            loop_detection=args.loop_detection,
            vocab_tree_path=args.vocab_tree,
            mapper_ba_use_gpu=args.mapper_ba_use_gpu,
            log_path=log_path,
        )
        return

    if args.cmd == "undistort":
        step_undistort(paths, log_path)
        return

    if args.cmd == "train":
        step_train(
            gs_repo=args.gs_repo,
            paths=paths,
            iterations=args.iterations,
            save_iters=args.save_iters,
            checkpoint_iters=args.checkpoint_iters,
            extra_args=args.extra,
            log_path=log_path,
            resume=args.resume,
        )
        return

    if args.cmd == "run":
        if args.video:
            step_extract(args.video, paths, args.fps, args.jpg_quality, args.skip_seconds, log_path)
        elif args.images_dir:
            ensure_dir(paths["images"])
            if not any(paths["images"].iterdir()):
                for pth in sorted(args.images_dir.glob("*")):
                    if pth.is_file():
                        shutil.copy2(str(pth), str(paths["images"] / pth.name))
        else:
            if not paths["images"].exists() or not any(paths["images"].iterdir()):
                raise ValueError("No input provided. Use --video or --images-dir or pre-fill workspace/images")

        step_colmap(
            paths=paths,
            camera_model=args.camera_model,
            single_camera=args.single_camera,
            use_gpu=args.use_gpu,
            matcher=args.matcher,
            overlap=args.overlap,
            loop_detection=args.loop_detection,
            vocab_tree_path=args.vocab_tree,
            mapper_ba_use_gpu=args.mapper_ba_use_gpu,
            log_path=log_path,
        )
        step_undistort(paths, log_path)
        step_train(
            gs_repo=args.gs_repo,
            paths=paths,
            iterations=args.iterations,
            save_iters=args.save_iters,
            checkpoint_iters=args.checkpoint_iters,
            extra_args=args.extra,
            log_path=log_path,
            resume=args.resume,
        )

        print("\n✅ DONE")
        print(f"Workspace: {paths['workspace']}")
        print(f"Log: {log_path}")
        print(f"Out: {paths['gs_out']}")
        return


if __name__ == "__main__":
    main()