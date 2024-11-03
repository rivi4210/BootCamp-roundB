import unittest
import numpy as np
import os
# from tests.server.logic_functions.stabilzation_frames import (
#     weighted_moving_average,
#     save_frame_to_folder,
#     fix_border,
#     detect_static_objects
# )
from server.server.logic_functions.aaa import (
    weighted_moving_average,
    save_frame_to_folder,
    fix_border,
    detect_static_objects
)


class TestWeightedMovingAverage(unittest.TestCase):
    # Existing tests for weighted_moving_average...

    def test_weighted_moving_average_positive_radius(self):
        curve = np.array([1, 2, 3, 4, 5])
        smoothed_curve = weighted_moving_average(curve, 1)
        self.assertEqual(len(smoothed_curve), 5, "Smoothed curve should have the same length as the input.")
        self.assertNotEqual(curve.tolist(), smoothed_curve.tolist(),
                            "Curve should be smoothed when radius is positive.")

    def test_weighted_moving_average_zero_radius(self):
        curve = np.array([1, 2, 3, 4, 5])
        smoothed_curve = weighted_moving_average(curve, 0)
        np.testing.assert_array_equal(curve, smoothed_curve, "Curve should be returned as is when radius is 0.")

    def test_weighted_moving_average_negative_radius(self):
        curve = np.array([1, 2, 3, 4, 5])
        smoothed_curve = weighted_moving_average(curve, -1)
        np.testing.assert_array_equal(curve, smoothed_curve, "Curve should be returned as is when radius is negative.")

    def test_weighted_moving_average_single_element(self):
        curve = np.array([10])
        smoothed_curve = weighted_moving_average(curve, 1)
        np.testing.assert_array_equal(curve, smoothed_curve, "Single element curve should remain unchanged.")

    def test_weighted_moving_average_empty_curve(self):
        curve = np.array([])
        smoothed_curve = weighted_moving_average(curve, 1)
        self.assertEqual(len(smoothed_curve), 0, "Empty curve should return an empty result.")

    def test_weighted_moving_average_large_radius(self):
        curve = np.array([1, 2, 3])
        smoothed_curve = weighted_moving_average(curve, 5)
        self.assertEqual(len(smoothed_curve), 3, "Curve should have the same length as the input.")
        self.assertNotEqual(curve.tolist(), smoothed_curve.tolist(),
                            "Smoothing should still be applied with large radius.")


class TestSaveFrameToFolder(unittest.TestCase):

    def test_save_frame_to_folder_creates_directory(self):
        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)  # Create a dummy frame
        test_folder = 'test_frames'
        save_frame_to_folder(test_frame, test_folder, 1)
        self.assertTrue(os.path.exists(test_folder), "Output folder should be created.")

    def test_save_frame_to_folder_saves_frame(self):
        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)  # Create a dummy frame
        test_folder = 'test_frames'
        save_frame_to_folder(test_frame, test_folder, 1)
        saved_frame_path = os.path.join(test_folder, "frame_0001.png")
        self.assertTrue(os.path.isfile(saved_frame_path), "Frame should be saved as an image file.")

    def tearDown(self):
        # Clean up the test directory after each test
        test_folder = 'test_frames'
        if os.path.exists(test_folder):
            for f in os.listdir(test_folder):
                os.remove(os.path.join(test_folder, f))
            os.rmdir(test_folder)


class TestFixBorder(unittest.TestCase):

    def test_fix_border_valid_frame(self):
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a valid frame
        fixed_frame = fix_border(test_frame)
        self.assertEqual(fixed_frame.shape, test_frame.shape,
                         "Fixed frame should have the same shape as the input frame.")

    def test_fix_border_invalid_frame(self):
        result = fix_border(None)
        self.assertIsNone(result, "Fixing border on None should return None.")

        result = fix_border(np.array([]))
        self.assertTrue(np.array_equal(result, np.array([])),
                        "Fixing border on an empty array should return an empty array.")


class TestDetectStaticObjects(unittest.TestCase):

    def test_no_static_points(self):
        # All points are moving faster than the camera
        prev_pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
        curr_pts = np.array([[15, 15], [25, 25]], dtype=np.float32)
        camera_transform = [1.0, 1.0, 0.1]

        # Expected: No points should be detected as static
        expected_static_prev = np.array([], dtype=np.float32).reshape(0, 2)
        expected_static_curr = np.array([], dtype=np.float32).reshape(0, 2)

        static_prev, static_curr = detect_static_objects(prev_pts, curr_pts, camera_transform, threshold=0.6)
        np.testing.assert_array_equal(static_prev, expected_static_prev, "Static previous points should be empty.")
        np.testing.assert_array_equal(static_curr, expected_static_curr, "Static current points should be empty.")

    def test_all_points_static(self):
        # All points have movement equal to or less than the camera
        prev_pts = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        curr_pts = np.array([[11, 10], [19, 20], [31, 30]], dtype=np.float32)
        camera_transform = [2.0, 1.5, 0.0]

        # Expected: All points detected as static
        expected_static_prev = prev_pts
        expected_static_curr = curr_pts

        static_prev, static_curr = detect_static_objects(prev_pts, curr_pts, camera_transform, threshold=1.5)
        np.testing.assert_array_equal(static_prev, expected_static_prev, "All points should be detected as static.")
        np.testing.assert_array_equal(static_curr, expected_static_curr, "All points should be detected as static.")


if __name__ == '__main__':
    unittest.main()
