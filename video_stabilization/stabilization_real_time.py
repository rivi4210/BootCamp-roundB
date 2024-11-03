import cv2
import numpy as np
import queue
import os
from collections import deque

# Initialize a global queue for stabilized frames
frame_queue = queue.Queue()
prev_frames_queue = deque(maxlen=5)


def save_frame_to_folder(frame, folder_path, frame_number):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist
    frame_filename = os.path.join(folder_path, f"frame_{frame_number:04d}.png")
    cv2.imwrite(frame_filename, frame)  # Save the frame as an image


def fix_border(frame):
    if frame is None or frame.size == 0:
        print("Error: Invalid frame passed to border fix.")
        return frame

    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.05)
    return cv2.warpAffine(frame, T, (s[1], s[0]))


def stabilize_video_sift_optical_flow(input_path, output_folder='stabilized_frames', smoothing_radius=50,
                                      motion_threshold=0.15, output_video_path='stabilized_output.mp4'):
    cap = cv2.VideoCapture(input_path)
    frame_number = 0

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        cap.release()
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(prev_gray, None)
    prev_frames_queue.append(prev_frame)

    transforms = []
    trajectory = []
    smoothed_trajectory = np.zeros((3,), np.float32)
    last_stabilized_frame = prev_frame.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_next, des_next = sift.detectAndCompute(gray, None)

        if des_prev is not None and des_next is not None:
            matches = bf.match(des_prev, des_next)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 4:
                print(f"Error: Not enough feature matches. Skipping frame {frame_number}.")
                stabilized_frame = last_stabilized_frame
            else:
                prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                next_pts = np.float32([kp_next[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Calculate optical flow for improved tracking
                next_pts_opt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, maxLevel=4)

                if next_pts_opt is None or status is None:
                    print(f"Optical flow failed at frame {frame_number}.")
                    stabilized_frame = last_stabilized_frame
                else:
                    good_prev_pts = prev_pts[status == 1]
                    good_next_pts = next_pts_opt[status == 1]

                    if len(good_prev_pts) >= 4 and len(good_next_pts) >= 4:
                        H, inliers = cv2.estimateAffinePartial2D(good_prev_pts, good_next_pts, method=cv2.RANSAC,
                                                                 ransacReprojThreshold=3)

                        if H is None:
                            print(f"Homography estimation failed at frame {frame_number}. Using last stabilized frame.")
                            stabilized_frame = last_stabilized_frame
                        else:
                            dx = H[0, 2]
                            dy = H[1, 2]
                            da = np.arctan2(H[1, 0], H[0, 0])

                            # Motion threshold to filter small jitters
                            if np.abs(dx) < motion_threshold:
                                dx = 0
                            if np.abs(dy) < motion_threshold:
                                dy = 0
                            if np.abs(da) < motion_threshold:
                                da = 0

                            # Add to transformations and trajectory lists
                            transform = np.array([dx, dy, da])
                            transforms.append(transform)
                            trajectory.append(transform)

                            # Smooth trajectory if enough frames
                            if len(transforms) > smoothing_radius:
                                smoothed_trajectory = np.mean(transforms[-smoothing_radius:], axis=0)

                            difference = smoothed_trajectory - np.sum(trajectory, axis=0)
                            dx += difference[0]
                            dy += difference[1]
                            da += difference[2]

                            # Apply the corrected transformation
                            H_corrected = np.array([
                                [np.cos(da), -np.sin(da), dx],
                                [np.sin(da), np.cos(da), dy]
                            ], dtype=np.float32)

                            # Combine transformations of multiple previous frames
                            for prev_frame in prev_frames_queue:
                                stabilized_frame = cv2.warpAffine(prev_frame, H_corrected,
                                                                  (prev_frame.shape[1], prev_frame.shape[0]))

                            stabilized_frame = fix_border(stabilized_frame)
                            last_stabilized_frame = stabilized_frame
                    else:
                        print(f"Insufficient valid points for transformation at frame {frame_number}.")
                        stabilized_frame = last_stabilized_frame
        else:
            print(f"Descriptors missing at frame {frame_number}. Using last stabilized frame.")
            stabilized_frame = last_stabilized_frame

        frame_queue.put(stabilized_frame)
        save_frame_to_folder(stabilized_frame, output_folder, frame_number)

        # Write the stabilized frame to the output video
        out.write(stabilized_frame)

        frame_number += 1

        # Update previous frame data
        prev_gray = gray
        kp_prev, des_prev = kp_next, des_next
        prev_frames_queue.append(frame)  # Add the current frame to the queue

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Function to retrieve stabilized frames from the queue
def get_stabilized_frame():
    try:
        if not frame_queue.empty():
            return frame_queue.get()
        else:
            print("Queue is empty.")
            return None
    except Exception as e:
        print(f"Error retrieving frame: {e}")
        return None


if __name__ == '__main__':
    stabilize_video_sift_optical_flow(
        'C:\\BootCamp\\project\\pythonProject\\video stabilization\\input2.mp4',
        output_video_path='stabilized_output2.mp4'
    )
