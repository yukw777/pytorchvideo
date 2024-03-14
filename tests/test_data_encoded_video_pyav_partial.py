import unittest
import pytest
import tempfile

import torch
from av.sidedata.sidedata import Type
from unittest.mock import Mock
from pytorchvideo.data.encoded_video_pyav_partial import (
    EncodedVideoPyAVPartial,
    _transform_mvs_to_tensor,
)
from utils import temp_encoded_video, temp_encoded_video_with_audio


class TestEncodedVideoPyAVPartial(unittest.TestCase):
    # Clip sampling is end time exclusive so we need to add _EPS to sample
    # all the frames of a video.
    _EPS = 1e-9

    def test_video_works(self):
        num_frames = 11
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (file_name, data):
            test_video = EncodedVideoPyAVPartial.from_path(
                file_name, decoder="pyav_partial"
            )
            self.assertAlmostEqual(test_video.duration, num_frames / fps)

            # All frames (0 - test_video.duration seconds)
            clip = test_video.get_clip(0, test_video.duration + self._EPS)
            _, _, height, width = data.size()
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            total_num_inter_frame = clip["gop_num_inter_frame"].sum()
            self.assertEqual(
                clip["motion_vector"].size(), (total_num_inter_frame, 4, height, width)
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame, 3, height, width)
            )
            self.assertEqual(clip["audio"], None)

            # Half frames
            clip = test_video.get_clip(0, test_video.duration / 2)
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            self.assertEqual(
                clip["motion_vector"].size(),
                (total_num_inter_frame // 2, 4, height, width),
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame // 2, 3, height, width)
            )
            self.assertEqual(clip["audio"], None)

            # We still return GOPs even when we ask for a clip beyond the duration,
            # since we return GOPs from the last previous key frame.
            clip = test_video.get_clip(test_video.duration + 1, test_video.duration + 3)
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            self.assertEqual(
                clip["motion_vector"].size(), (total_num_inter_frame, 4, height, width)
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame, 3, height, width)
            )
            self.assertEqual(clip["audio"], None)
            test_video.close()

    def test_video_with_shorter_audio_works(self):
        num_audio_samples = 8000
        num_frames = 11
        fps = 5
        audio_rate = 8000
        with temp_encoded_video_with_audio(
            num_frames=num_frames,
            fps=fps,
            num_audio_samples=num_audio_samples,
            audio_rate=audio_rate,
        ) as (file_name, video_data, audio_data):
            test_video = EncodedVideoPyAVPartial.from_path(
                file_name, decoder="pyav_partial"
            )

            # Duration is max of both streams, therefore, the video duration will be expected.
            self.assertAlmostEqual(test_video.duration, num_frames / fps)

            # All audio (0 - 1 seconds)
            clip = test_video.get_clip(0, test_video.duration + self._EPS)
            _, _, height, width = video_data.size()
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            total_num_inter_frame = clip["gop_num_inter_frame"].sum()
            self.assertEqual(
                clip["motion_vector"].size(), (total_num_inter_frame, 4, height, width)
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame, 3, height, width)
            )
            self.assertTrue(clip["audio"].equal(audio_data))

            # Half frames
            clip = test_video.get_clip(0, test_video.duration / 2)
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            self.assertEqual(
                clip["motion_vector"].size(),
                # we subtract 1 here due to the quirk in temp_encoded_video_with_audio()
                (total_num_inter_frame // 2 - 1, 4, height, width),
            )
            self.assertEqual(
                clip["residual"].size(),
                # we subtract 1 here due to the quirk in temp_encoded_video_with_audio()
                (total_num_inter_frame // 2 - 1, 3, height, width),
            )
            self.assertTrue(clip["audio"].equal(audio_data))

            test_video.close()

    def test_video_with_longer_audio_works(self):
        audio_rate = 10000
        fps = 5
        num_frames = 5
        num_audio_samples = 50000
        with temp_encoded_video_with_audio(
            num_frames=num_frames,
            fps=fps,
            num_audio_samples=num_audio_samples,
            audio_rate=audio_rate,
        ) as (file_name, video_data, audio_data):
            test_video = EncodedVideoPyAVPartial.from_path(
                file_name, decoder="pyav_partial"
            )

            # All audio
            clip = test_video.get_clip(0, test_video.duration + self._EPS)
            _, _, height, width = video_data.size()
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            total_num_inter_frame = clip["gop_num_inter_frame"].sum()
            self.assertEqual(
                clip["motion_vector"].size(), (total_num_inter_frame, 4, height, width)
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame, 3, height, width)
            )
            self.assertTrue(clip["audio"].equal(audio_data))

            # No frames (3 - 5 seconds)
            clip = test_video.get_clip(test_video.duration + 1, test_video.duration + 2)
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            self.assertEqual(
                clip["motion_vector"].size(), (total_num_inter_frame, 4, height, width)
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame, 3, height, width)
            )
            self.assertEqual(clip["audio"], None)

            test_video.close()

    def test_decode_audio_is_false(self):
        audio_rate = 10000
        fps = 5
        num_frames = 5
        num_audio_samples = 50000
        with temp_encoded_video_with_audio(
            num_frames=num_frames,
            fps=fps,
            num_audio_samples=num_audio_samples,
            audio_rate=audio_rate,
        ) as (file_name, video_data, _):
            test_video = EncodedVideoPyAVPartial.from_path(
                file_name, decode_audio=False, decoder="pyav_partial"
            )

            # All audio
            clip = test_video.get_clip(0, test_video.duration + self._EPS)
            _, _, height, width = video_data.size()
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            total_num_inter_frame = clip["gop_num_inter_frame"].sum()
            self.assertEqual(
                clip["motion_vector"].size(), (total_num_inter_frame, 4, height, width)
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame, 3, height, width)
            )
            self.assertEqual(clip["audio"], None)

            test_video.close()

    def test_file_api(self):
        num_frames = 11
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (file_name, data):
            with open(file_name, "rb") as f:
                test_video = EncodedVideoPyAVPartial(f)

            self.assertAlmostEqual(test_video.duration, num_frames / fps)
            clip = test_video.get_clip(0, test_video.duration + self._EPS)
            _, _, height, width = data.size()
            self.assertEqual(
                clip["key_frame"].size(),
                (len(clip["gop_num_inter_frame"]), 3, height, width),
            )
            total_num_inter_frame = clip["gop_num_inter_frame"].sum()
            self.assertEqual(
                clip["motion_vector"].size(), (total_num_inter_frame, 4, height, width)
            )
            self.assertEqual(
                clip["residual"].size(), (total_num_inter_frame, 3, height, width)
            )
            self.assertEqual(clip["audio"], None)

    def test_open_video_failure(self):
        with pytest.raises(FileNotFoundError):
            test_video = EncodedVideoPyAVPartial.from_path("non_existent_file.txt")
            test_video.close()

    def test_decode_video_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"This is not an mp4 file")
            with pytest.raises(RuntimeError):
                test_video = EncodedVideoPyAVPartial.from_path(f.name)
                test_video.close()

    def test_transform_mvs_to_tensor(self):
        gop_inter_frames = [
            [
                Mock(
                    pts=0,
                    height=32,
                    width=64,
                    side_data={
                        Type.MOTION_VECTORS: [
                            Mock(
                                w=16,
                                h=16,
                                dst_x=8,
                                dst_y=8,
                                src_x=10,
                                src_y=9,
                                motion_x=8,
                                motion_y=4,
                                motion_scale=4,
                                source=-1,
                            )
                        ]
                    },
                )
            ],
            [
                Mock(
                    pts=1,
                    height=32,
                    width=64,
                    side_data={
                        Type.MOTION_VECTORS: [
                            Mock(
                                w=4,
                                h=4,
                                dst_x=18,
                                dst_y=18,
                                src_x=17,
                                src_y=15,
                                motion_x=4,
                                motion_y=12,
                                motion_scale=4,
                                source=-1,
                            )
                        ]
                    },
                )
            ],
            [
                Mock(
                    pts=2,
                    height=32,
                    width=64,
                    side_data={
                        Type.MOTION_VECTORS: [
                            Mock(
                                w=16,
                                h=16,
                                dst_x=8,
                                dst_y=8,
                                src_x=10,
                                src_y=9,
                                motion_x=8,
                                motion_y=4,
                                motion_scale=4,
                                source=1,
                            )
                        ]
                    },
                )
            ],
            [
                Mock(
                    pts=3,
                    height=32,
                    width=64,
                    side_data={
                        Type.MOTION_VECTORS: [
                            Mock(
                                w=4,
                                h=4,
                                dst_x=18,
                                dst_y=18,
                                src_x=17,
                                src_y=15,
                                motion_x=4,
                                motion_y=12,
                                motion_scale=4,
                                source=1,
                            )
                        ]
                    },
                )
            ],
            [
                Mock(
                    pts=4,
                    height=32,
                    width=64,
                    side_data={
                        Type.MOTION_VECTORS: [
                            Mock(
                                w=8,
                                h=16,
                                dst_x=12,
                                dst_y=8,
                                src_x=13,
                                src_y=12,
                                motion_x=4,
                                motion_y=16,
                                motion_scale=4,
                                source=-1,
                            ),
                            Mock(
                                w=16,
                                h=8,
                                dst_x=24,
                                dst_y=4,
                                src_x=14,
                                src_y=12,
                                motion_x=-40,
                                motion_y=32,
                                motion_scale=4,
                                source=-1,
                            ),
                        ]
                    },
                ),
                Mock(
                    pts=5,
                    height=32,
                    width=64,
                    side_data={
                        Type.MOTION_VECTORS: [
                            Mock(
                                w=8,
                                h=4,
                                dst_x=12,
                                dst_y=8,
                                src_x=16,
                                src_y=10,
                                motion_x=16,
                                motion_y=8,
                                motion_scale=4,
                                source=1,
                            ),
                            Mock(
                                w=4,
                                h=8,
                                dst_x=6,
                                dst_y=8,
                                src_x=16,
                                src_y=10,
                                motion_x=40,
                                motion_y=8,
                                motion_scale=4,
                                source=1,
                            ),
                        ]
                    },
                ),
            ],
        ]
        gop_0_inter_frame_0 = torch.zeros(4, 32, 64, dtype=torch.int)
        gop_0_inter_frame_0[0, 0:16, 0:16] = 2
        gop_0_inter_frame_0[1, 0:16, 0:16] = 1

        gop_1_inter_frame_0 = torch.zeros(4, 32, 64, dtype=torch.int)
        gop_1_inter_frame_0[0, 16:20, 16:20] = -1
        gop_1_inter_frame_0[1, 16:20, 16:20] = -3

        gop_2_inter_frame_0 = torch.zeros(4, 32, 64, dtype=torch.int)
        gop_2_inter_frame_0[2, 0:16, 0:16] = 2
        gop_2_inter_frame_0[3, 0:16, 0:16] = 1

        gop_3_inter_frame_0 = torch.zeros(4, 32, 64, dtype=torch.int)
        gop_3_inter_frame_0[2, 16:20, 16:20] = -1
        gop_3_inter_frame_0[3, 16:20, 16:20] = -3

        gop_4_inter_frame_0 = torch.zeros(4, 32, 64, dtype=torch.int)
        gop_4_inter_frame_0[0, 0:16, 8:16] = 1
        gop_4_inter_frame_0[1, 0:16, 8:16] = 4
        gop_4_inter_frame_0[0, 0:8, 16:32] = -10
        gop_4_inter_frame_0[1, 0:8, 16:32] = 8
        gop_4_inter_frame_1 = torch.zeros(4, 32, 64, dtype=torch.int)
        gop_4_inter_frame_1[2, 6:10, 8:16] = 4
        gop_4_inter_frame_1[3, 6:10, 8:16] = 2
        gop_4_inter_frame_1[2, 4:12, 4:8] = 10
        gop_4_inter_frame_1[3, 4:12, 4:8] = 2

        expected_gop_inter_frame_tensors = [
            [(gop_0_inter_frame_0, 0)],
            [(gop_1_inter_frame_0, 1)],
            [(gop_2_inter_frame_0, 2)],
            [(gop_3_inter_frame_0, 3)],
            [(gop_4_inter_frame_0, 4), (gop_4_inter_frame_1, 5)],
        ]
        for inter_frame_tensors, expected_inter_frame_tensors in zip(
            _transform_mvs_to_tensor(gop_inter_frames),
            expected_gop_inter_frame_tensors,
            strict=True,
        ):
            for (inter_frame_tensor, pts), (
                expected_inter_frame_tensor,
                expected_pts,
            ) in zip(inter_frame_tensors, expected_inter_frame_tensors, strict=True):
                self.assertTrue(inter_frame_tensor.equal(expected_inter_frame_tensor))
                self.assertEqual(pts, expected_pts)
