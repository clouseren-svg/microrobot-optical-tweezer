"""
Detection Demo
==============
Demonstrates microrobot detection and ellipse/trapping-point detection
on synthetic microscope images.

Usage:
    python demo/run_detection_demo.py --num_robots 3 --save_viz
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import cv2
import numpy as np

from simulation.microscope_simulator import MicroscopeSimulator
from detection.ellipse_detector import ArcSupportEllipseDetector


def main():
    parser = argparse.ArgumentParser(
        description="Microrobot detection demo")
    parser.add_argument("--num_robots", type=int, default=3,
                        help="Number of microrobots in scene")
    parser.add_argument("--save_viz", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--output_dir", type=str, default="data/synthetic",
                        help="Output directory for visualizations")
    args = parser.parse_args()

    print("=" * 60)
    print("Microrobot Detection Demo (MARSS 2022)")
    print("=" * 60)

    # 1. Generate synthetic image
    print("\n[1/3] Generating synthetic microscope image...")
    sim = MicroscopeSimulator(image_size=(640, 480))
    image, annotations = sim.generate_multiple(
        num_robots=args.num_robots,
        return_annotations=True
    )
    print(f"      Generated image: {image.shape}")
    if annotations:
        print(f"      Ground-truth robots: {len(annotations['boxes'])}")

    # 2. Detect ellipses / trapping points
    print("\n[2/3] Running ellipse detection (arc-support + k-means)...")
    detector = ArcSupportEllipseDetector(kmeans_k=min(13, args.num_robots + 2))
    ellipses, debug_img = detector.detect(image, return_debug=True)
    trapping_points = detector.get_trapping_points(ellipses)

    print(f"      Ellipses detected: {len(ellipses)}")
    print(f"      Trapping points: {len(trapping_points)}")

    for i, ell in enumerate(ellipses):
        print(f"      Ellipse {i+1}: center=({ell.center[0]:.1f}, {ell.center[1]:.1f}), "
              f"axes=({ell.axes[0]:.1f}, {ell.axes[1]:.1f}), "
              f"score={ell.score:.3f}")

    # 3. Visualization
    print("\n[3/3] Creating visualization...")
    vis = image.copy()

    # Draw ground truth boxes
    if annotations:
        for box in annotations['boxes']:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, "GT", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw detected ellipses and trapping points
    for ell in ellipses:
        cx, cy = int(ell.center[0]), int(ell.center[1])
        ax1, ax2 = int(ell.axes[0] / 2), int(ell.axes[1] / 2)
        cv2.ellipse(vis, (cx, cy), (ax1, ax2), int(ell.angle),
                   0, 360, (0, 0, 255), 2)

    for tp in trapping_points:
        tpx, tpy = int(tp[0]), int(tp[1])
        cv2.circle(vis, (tpx, tpy), 5, (255, 0, 0), -1)
        cv2.circle(vis, (tpx, tpy), 7, (255, 255, 255), 1)

    # Metrics overlay
    if annotations:
        gt_count = len(annotations['boxes'])
        det_count = len(ellipses)
        text = f"GT: {gt_count} | Detected: {det_count} | Trapping Points: {len(trapping_points)}"
    else:
        text = f"Detected: {len(ellipses)} | Trapping Points: {len(trapping_points)}"

    cv2.putText(vis, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Save results
    if args.save_viz:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "detection_demo.jpg")
        cv2.imwrite(out_path, vis)
        print(f"\n      Visualization saved to: {out_path}")

        if debug_img is not None:
            debug_path = os.path.join(args.output_dir, "detection_debug.jpg")
            cv2.imwrite(debug_path, debug_img)
            print(f"      Debug image saved to: {debug_path}")
    else:
        # Display
        cv2.imshow("Detection Demo", vis)
        print("\n      Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
