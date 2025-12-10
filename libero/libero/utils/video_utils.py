import os
import imageio
import numpy as np


class VideoWriter:
    def __init__(self, video_path, save_video=False, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        # Buffers keyed by (stream_name, idx)
        self.image_buffer = {}
        self.last_images = {}
        self.single_video = single_video

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def _ensure_keys(self, stream, idx):
        key = (stream, idx)
        if key not in self.image_buffer:
            self.image_buffer[key] = []
        if key not in self.last_images:
            self.last_images[key] = None
        return key

    def _overlay_if_done(self, frame, last_frame, done):
        if not done:
            return frame, last_frame
        if last_frame is None:
            last_frame = frame
        original_image = np.copy(last_frame)
        blank_image = np.ones_like(original_image) * 128
        blank_image[:, :, 0] = 0
        blank_image[:, :, -1] = 0
        transparency = 0.7
        original_image = (
            original_image * (1 - transparency) + blank_image * transparency
        )
        return original_image.astype(np.uint8), last_frame

    def append_image(self, img, idx=0, stream="default"):
        """Directly append an image to a stream."""
        if self.save_video:
            key = self._ensure_keys(stream, idx)
            self.image_buffer[key].append(img)

    def append_obs(self, obs, done, idx=0, camera_name="agentview_image", stream_name=None):
        """Append a camera observation to the video."""
        if self.save_video:
            stream = stream_name or camera_name
            key = self._ensure_keys(stream, idx)
            frame = obs[camera_name][::-1]
            frame, self.last_images[key] = self._overlay_if_done(
                frame, self.last_images[key], done
            )
            self.image_buffer[key].append(frame)

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_vector_obs(self, obs, dones, camera_name="agentview_image", stream_name=None):
        if self.save_video:
            for i in range(len(obs)):
                self.append_obs(obs[i], dones[i], i, camera_name, stream_name)

    def append_vector_combined(self, obs, dones, left_key="agentview_image", right_key="robot0_eye_in_hand_image", stream_name="combined"):
        if not self.save_video:
            return
        for i in range(len(obs)):
            stream_key = stream_name or "combined"
            buffer_key = self._ensure_keys(stream_key, i)
            left = obs[i][left_key][::-1]
            right = obs[i][right_key][::-1]
            combined = np.concatenate([left, right], axis=1)
            combined, self.last_images[buffer_key] = self._overlay_if_done(
                combined, self.last_images[buffer_key], dones[i]
            )
            self.image_buffer[buffer_key].append(combined)

    def save(self):
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                # One video per stream, merged across env indices
                streams = set(stream for stream, _ in self.image_buffer.keys())
                for stream in streams:
                    video_name = os.path.join(self.video_path, f"{stream}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for (s, idx) in sorted(self.image_buffer.keys(), key=lambda x: (x[0], x[1])):
                        if s != stream:
                            continue
                        for im in self.image_buffer[(s, idx)]:
                            video_writer.append_data(im)
                    video_writer.close()
            else:
                for stream, idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{stream}_{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[(stream, idx)]:
                        video_writer.append_data(im)
                    video_writer.close()
            print(f"Saved videos to {self.video_path}.")
