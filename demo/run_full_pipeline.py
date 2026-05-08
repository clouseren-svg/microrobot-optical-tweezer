"""
Full Pipeline Demo
==================
Demonstrates the complete automatic trapping pipeline:
    1. Initialize simulated hardware
    2. Detect & track microrobots
    3. Move laser spots to trap robots
    4. Verify trapping success

Usage:
    python demo/run_full_pipeline.py --num_robots 3 --visualize
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import cv2
import time

from control.trapping_controller import AutomaticTrappingController, TrappingState


def main():
    parser = argparse.ArgumentParser(
        description="Full automatic trapping pipeline demo")
    parser.add_argument("--num_robots", type=int, default=3,
                        help="Target number of robots to trap")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Timeout in seconds")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live visualization")
    parser.add_argument("--save_video", type=str, default="",
                        help="Save demo video to file")
    args = parser.parse_args()

    print("=" * 70)
    print("Full Automatic Trapping Pipeline Demo")
    print("Paper: Machine Learning-Based Real-Time Localization and")
    print("       Automatic Trapping of Multiple Microrobots in Optical Tweezer")
    print("       (MARSS 2022)")
    print("=" * 70)

    # Initialize controller
    print("\n[Init] Creating controller with simulation hardware...")
    controller = AutomaticTrappingController(
        hardware_mode="simulation",
        num_traps=5,
        detection_interval=3,
        trapping_threshold=10.0
    )

    if not controller.connect():
        print("[Error] Failed to connect to hardware")
        return

    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, 20.0, (640, 480))

    # Run trapping sequence
    print(f"\n[Run] Target: trap {args.num_robots} robots")
    print(f"       Timeout: {args.timeout}s")
    print(f"       Visualization: {'ON' if args.visualize else 'OFF'}")
    print("-" * 70)

    start_time = time.time()
    step_count = 0
    result = None

    try:
        while time.time() - start_time < args.timeout:
            result = controller.step()
            step_count += 1

            if step_count % 10 == 0:
                print(f"  Step {step_count:4d} | {result.message}")

            if result.success:
                print("\n[SUCCESS] All target robots trapped!")
                break

            # Visualization
            if args.visualize or args.save_video:
                vis = controller.get_visualization()
                if vis is not None:
                    if args.visualize:
                        cv2.imshow("Auto Trapping", vis)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\n[User] Interrupted by user")
                            break
                    if video_writer:
                        video_writer.write(vis)

            time.sleep(0.03)

        else:
            print("\n[Timeout] Trapping sequence timed out")

    except KeyboardInterrupt:
        print("\n[User] Interrupted by user")

    finally:
        # Final state
        final_vis = controller.get_visualization()
        if final_vis is not None and args.save_video:
            for _ in range(30):  # 1 second final frame
                video_writer.write(final_vis)

        if video_writer:
            video_writer.release()
            print(f"\n[Save] Video saved to: {args.save_video}")

        controller.disconnect()

        print("\n" + "=" * 70)
        print("Pipeline Demo Summary")
        print("=" * 70)
        print(f"  Total steps: {step_count}")
        print(f"  Final state: {controller.state.value}")
        if result is not None:
            print(f"  Robots detected: {result.num_robots_detected}")
            print(f"  Robots trapped: {result.num_robots_trapped}")
            print(f"  Success: {result.success}")
        else:
            print(f"  No result available")
        print("=" * 70)

        if args.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
