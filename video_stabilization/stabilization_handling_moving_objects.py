import cv2
import numpy as np


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]


def smooth_trajectory(trajectory, radius):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(trajectory.shape[1]):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed_trajectory


def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.02)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def detect_static_objects(prev_pts, curr_pts, camera_transform, threshold=2.0):
    dx_camera, dy_camera, da_camera = camera_transform

    object_displacements = np.linalg.norm(curr_pts - prev_pts, axis=1)

    camera_displacement = np.linalg.norm([dx_camera, dy_camera])

    static_points_prev = []
    static_points_curr = []

    for i, displacement in enumerate(object_displacements):
        if abs(displacement) <= abs(camera_displacement):
            static_points_prev.append(prev_pts[i])
            static_points_curr.append(curr_pts[i])
    return np.array(static_points_prev), np.array(static_points_curr)


def stabilize_video(input_path, output_path, smoothing_radius=50):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    sift = cv2.SIFT_create()
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(prev_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 50:
            print(f"Frame {i}: Not enough matches found ({len(matches)}). Skipping this frame.")
            continue

        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts)
        prev_pts = prev_pts[status == 1]
        curr_pts = curr_pts[status == 1]

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is None:
            print(f"Frame {i}: Could not find a valid transformation matrix. Skipping this frame.")
            continue

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms[i] = [dx, dy, da]
        print(transforms[i])

        statics_obj_prev, statics_obj_curr = detect_static_objects(prev_pts, curr_pts, transforms[i])
        m2, _ = cv2.estimateAffinePartial2D(statics_obj_prev, statics_obj_curr)
        dx2 = m2[0, 2]
        dy2 = m2[1, 2]
        da2 = np.arctan2(m2[1, 0], m2[0, 0])
        transforms[i] = [dx2, dy2, da2]

        prev_gray = curr_gray
        prev_kp = curr_kp
        prev_des = curr_des

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, smoothing_radius)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames - 2):
        success, frame = cap.read()
        if not success:
            break

        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fix_border(frame_stabilized)

        out.write(frame_stabilized)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    stabilize_video('input4.mp4', 'stabilized_output_video_without_moving_obj.mp4')
