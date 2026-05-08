"""
Unit Tests for Ellipse Detection
=================================
Tests the arc-support ellipse detector on synthetic images
with known ground-truth ellipses.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import cv2
import numpy as np

from detection.ellipse_detector import ArcSupportEllipseDetector, detect_ellipses


class TestEllipseDetection(unittest.TestCase):

    def setUp(self):
        self.detector = ArcSupportEllipseDetector(
            canny_low=50,
            canny_high=150,
            kmeans_k=2,
            min_ellipse_axis=10,
            max_ellipse_axis=150
        )

    def _draw_ellipse(self, img, center, axes, angle, color=180):
        """Draw a filled ellipse on image."""
        cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)

    def test_single_ellipse(self):
        """Detect a single clean ellipse."""
        img = np.zeros((300, 300), dtype=np.uint8)
        gt_center = (150, 150)
        gt_axes = (60, 40)
        gt_angle = 30
        self._draw_ellipse(img, gt_center, gt_axes, gt_angle)

        ellipses, debug = self.detector.detect(img, return_debug=True)

        self.assertGreaterEqual(len(ellipses), 1,
                                "Should detect at least one ellipse")

        best = max(ellipses, key=lambda e: e.score)
        dist = np.sqrt((best.center[0] - gt_center[0])**2 +
                       (best.center[1] - gt_center[1])**2)

        self.assertLess(dist, 15, f"Center error too large: {dist:.1f}px")
        print(f"  [PASS] Single ellipse: center_err={dist:.1f}px")

    def test_multiple_ellipses(self):
        """Detect multiple ellipses in one image."""
        img = np.zeros((400, 500), dtype=np.uint8)

        ellipses_gt = [
            ((100, 100), (40, 30), 0),
            ((350, 120), (50, 35), 45),
            ((200, 300), (45, 25), -30),
        ]

        for center, axes, angle in ellipses_gt:
            self._draw_ellipse(img, center, axes, angle,
                               color=np.random.randint(150, 200))

        # Use higher k for multiple ellipses
        detector = ArcSupportEllipseDetector(kmeans_k=len(ellipses_gt))
        detected, _ = detector.detect(img)

        self.assertGreaterEqual(len(detected), 2,
                                f"Should detect most ellipses, got {len(detected)}")
        print(f"  [PASS] Multiple ellipses: detected {len(detected)}/{len(ellipses_gt)}")

    def test_noisy_ellipse(self):
        """Detect ellipse with additive noise."""
        img = np.zeros((300, 300), dtype=np.uint8)
        self._draw_ellipse(img, (150, 150), (50, 35), 20)

        # Add noise
        noise = np.random.normal(0, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        ellipses, _ = self.detector.detect(img)
        self.assertGreaterEqual(len(ellipses), 1, "Should detect ellipse in noisy image")
        print(f"  [PASS] Noisy ellipse: detected {len(ellipses)} ellipse(s)")

    def test_empty_image(self):
        """Handle image with no ellipses gracefully."""
        img = np.zeros((200, 200), dtype=np.uint8)
        ellipses, _ = self.detector.detect(img)
        self.assertEqual(len(ellipses), 0, "Should return empty list for empty image")
        print(f"  [PASS] Empty image: no detections as expected")

    def test_trapping_points(self):
        """Verify trapping point extraction."""
        img = np.zeros((300, 300), dtype=np.uint8)
        self._draw_ellipse(img, (150, 150), (50, 40), 0)

        ellipses, _ = self.detector.detect(img)
        self.assertGreater(len(ellipses), 0)

        points = self.detector.get_trapping_points(ellipses)
        self.assertEqual(len(points), len(ellipses),
                         "Should return one trapping point per ellipse")

        for i, (px, py) in enumerate(points):
            ex, ey = ellipses[i].center
            dist = np.sqrt((px - ex)**2 + (py - ey)**2)
            self.assertLess(dist, 5, "Trapping point should be near ellipse center")

        print(f"  [PASS] Trapping points: {len(points)} points near centers")


class TestEllipseDetectorBenchmark(unittest.TestCase):
    """Benchmark detection speed."""

    def test_speed(self):
        """Measure detection FPS on synthetic image."""
        import time

        img = np.zeros((480, 640), dtype=np.uint8)
        cv2.ellipse(img, (320, 240), (80, 60), 30, 0, 360, 180, -1)
        cv2.ellipse(img, (150, 150), (40, 30), 0, 0, 360, 160, -1)

        detector = ArcSupportEllipseDetector(kmeans_k=2)

        num_runs = 10
        start = time.time()
        for _ in range(num_runs):
            detector.detect(img)
        elapsed = time.time() - start

        fps = num_runs / elapsed
        print(f"\n  [BENCHMARK] Ellipse detection speed: {fps:.1f} FPS")
        self.assertGreater(fps, 5, "Detection should run at reasonable speed")


def run_tests():
    print("=" * 60)
    print("Ellipse Detection Unit Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestEllipseDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestEllipseDetectorBenchmark))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
