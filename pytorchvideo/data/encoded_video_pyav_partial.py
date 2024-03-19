import logging
import math
from fractions import Fraction
from typing import BinaryIO, Any

import av
import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from .utils import pts_to_secs, secs_to_pts


logger = logging.getLogger(__name__)


class EncodedVideoPyAVPartial(EncodedVideo):
    """
    EncodedVideoPyAVPartial is an abstraction for accessing clips from an encoded video
    using a custom build PyAV as the "partial" decoding backend. It supports selective
    decoding when header information is available.
    """

    def __init__(
        self,
        file: BinaryIO,
        video_name: str | None = None,
        decode_video: bool = True,
        decode_audio: bool = True,
        perform_seek: bool = True,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
            perform_seek:
                Whether or not to seek time to the underlying video container. Only affects
                the audio stream as we always perform a seek for the video stream.

                NOTE: seeks may be slow on larger files, e.g. on a networked filesystem
        """
        self.perform_seek = perform_seek
        self._video_name = video_name
        self._decode_video = decode_video
        self._decode_audio = decode_audio

        try:
            self._container = av.open(file)
        except Exception as e:
            raise RuntimeError(f"Failed to open video {video_name}. {e}") from e

        if self._container is None or len(self._container.streams.video) == 0:
            raise RuntimeError(f"Video stream not found {video_name}")

        # Retrieve video header information if available.
        video_stream = self._container.streams.video[0]
        self._video_time_base = video_stream.time_base
        self._video_start_pts = video_stream.start_time
        if self._video_start_pts is None:
            self._video_start_pts = 0.0

        video_duration = video_stream.duration

        # Retrieve audio header information if available.
        audio_duration = None
        self._has_audio = None
        if self._decode_audio:
            self._has_audio = self._container.streams.audio
            if self._has_audio:
                self._audio_time_base = self._container.streams.audio[0].time_base
                self._audio_start_pts = self._container.streams.audio[0].start_time
                if self._audio_start_pts is None:
                    self._audio_start_pts = 0.0

                audio_duration = self._container.streams.audio[0].duration

        # If duration isn't found in header the whole video is decoded to
        # determine the duration.
        # NOTE: We don't store the decoded frames here like EncodedVideoPyAV
        # since we always perform seek in order to find the appropriate key frames.
        video, audio = (None, None)
        if audio_duration is None and video_duration is None:
            video, audio = self._pyav_decode_video()
            if video is None:
                raise RuntimeError("Unable to decode video stream")

            # take the pts of the last inter frame of the last GOP
            video_duration = video["motion_vectors_and_pts"][-1][-1][1]
            if audio is not None:
                audio_duration = audio[-1][1]

        # Take the largest duration of either video or duration stream.
        if audio_duration is not None and video_duration is not None:
            self._duration = max(
                pts_to_secs(
                    video_duration, self._video_time_base, self._video_start_pts
                ),
                pts_to_secs(
                    audio_duration, self._audio_time_base, self._audio_start_pts
                ),
            )
        elif video_duration is not None:
            self._duration = pts_to_secs(
                video_duration, self._video_time_base, self._video_start_pts
            )

        elif audio_duration is not None:
            self._duration = pts_to_secs(
                audio_duration, self._audio_time_base, self._audio_start_pts
            )

    @property
    def rate(self) -> str | Fraction:
        """
        Returns:
            rate: the frame rate of the video
        """
        return self._container.streams.video[0].rate

    @property
    def bit_rate(self) -> int:
        """
        Returns:
            bit_rate: the bit rate of the underlying video
        """
        return self._container.streams.video[0].bit_rate

    @property
    def pix_fmt(self) -> int:
        """
        Returns:
            pix_fmt: the pixel format of the underlying video
        """
        return self._container.streams.video[0].pix_fmt

    @property
    def name(self) -> str | None:
        """
        Returns:
            name: the name of the stored video if set.
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> dict[str, torch.Tensor | None]:
        """
        Retrieves partially decoded frames from the encoded video grouped as GOPs.
        As each GOP starts with a key frame, the retrieved frames may not align
        exactly with the given start time in seconds, as the key frame may be located
        before the start time. Note that 1) if you want to avoid float precision issue
        and need accurate frames, please use Fraction for start_sec and end_sec.
        2) As end_sec is exclusive, so you may need to use
        `get_clip(start_sec, duration + EPS)` to get the last frame.

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_data:
                A dictionary mapping the entries at "keyframe", "motion_vector",
                "residual" and "audio" to tensors.

                "key_frame": A tensor of the clip's RGB key frames with shape:
                (num_gop, channel, height, width). The frames are of type torch.uint8.

                "motion_vector": A tensor of the motion vectors for the GOPs with shape:
                (num_inter_frame, channel, height, width). The motion vector values are of type
                torch.float32. The channel number is 4. The first 2 channels are for the
                x and y values for forward prediction (list0) and the last 2 are for backward
                prediction (list1). Note that the channels for backward prediction may be
                all 0 for forward prediction only frames, e.g., P-frames.

                "residual": A tensor of the residuals for the GOPs with shape:
                (num_inter_frame, channel, height, width). These are RGB frames and the tensor
                is of type torch.uint8.

                "gop_num_inter_frame": A tensor of the number of inter frames for each GOP
                with shape (num_gop). This can be used to split the motion vector and
                residual tensor based on their GOPs. Note that the number of inter frames
                for a GOP is one fewer than the length of the GOP as it excludes the key frame.

                "audio": A tensor of the clip's audio samples with shape:
                (samples). The samples are of type torch.uint8.

            Returns None if no video or audio found within time range.

        """
        video, audio = self._pyav_decode_video(start_sec, end_sec)

        key_frame: torch.Tensor | None = None
        motion_vector: torch.Tensor | None = None
        residual: torch.Tensor | None = None
        gop_num_inter_frame: torch.Tensor | None = None
        if video is not None:
            key_frame = torch.stack(
                [f for f, _ in video["key_frames_and_pts"]]
            ).permute(0, 3, 1, 2)
            motion_vector = torch.stack(
                [mv for mvs_pts in video["motion_vectors_and_pts"] for mv, _ in mvs_pts]
            )
            residual = torch.stack(
                [res for res_pts in video["residuals_and_pts"] for res, _ in res_pts]
            ).permute(0, 3, 1, 2)
            gop_num_inter_frame = video["gop_num_inter_frame"]

        audio_samples = None
        if self._has_audio and audio is not None:
            audio_start_pts = secs_to_pts(
                start_sec,
                self._audio_time_base,
                self._audio_start_pts,
                round_mode="ceil",
            )
            audio_end_pts = secs_to_pts(
                end_sec,
                self._audio_time_base,
                self._audio_start_pts,
                round_mode="ceil",
            )
            audio_samples = torch.cat(
                [
                    f
                    for f, pts in audio
                    if pts >= audio_start_pts and pts < audio_end_pts
                ]
            )

        if key_frame is None:
            logger.debug(
                f"No GOPs found within {start_sec} and {end_sec} seconds. "
                f"Video starts at time 0 and ends at {self.duration}."
            )

        return {
            "key_frame": key_frame,
            "motion_vector": motion_vector,
            "residual": residual,
            "gop_num_inter_frame": gop_num_inter_frame,
            "audio": audio_samples,
        }

    def close(self):
        """
        Closes the internal video container.
        """
        if self._container is not None:
            self._container.close()

    def _pyav_decode_video(
        self, start_secs: float = 0.0, end_secs: float = math.inf
    ) -> tuple[dict[str, Any], list[tuple[torch.Tensor, int]] | None]:
        """
        Selectively decodes a video between start_pts and end_pts in time units of the
        self._video's timebase.
        """
        key_frames_and_pts: list[tuple[torch.Tensor, int]] | None = None
        motion_vectors_and_pts: list[list[tuple[torch.Tensor, int]]] | None = None
        residuals_and_pts: list[list[tuple[torch.Tensor, int]]] | None = None
        gop_num_inter_frame: torch.Tensor | None = None
        audio_and_pts: list[tuple[torch.Tensor, int]] | None = None
        try:
            if self._decode_video:
                pyav_video_key_frames, pyav_video_inter_frames, _ = (
                    _pyav_partial_decode_video_stream(
                        self._container,
                        secs_to_pts(
                            start_secs,
                            self._video_time_base,
                            self._video_start_pts,
                            round_mode="ceil",
                        ),
                        secs_to_pts(
                            end_secs,
                            self._video_time_base,
                            self._video_start_pts,
                            round_mode="ceil",
                        ),
                    )
                )
                if len(pyav_video_key_frames) > 0:
                    key_frames_and_pts = [
                        (torch.from_numpy(frame.to_rgb().to_ndarray()), frame.pts)
                        for frame in pyav_video_key_frames
                    ]
                    motion_vectors_and_pts = _transform_mvs_to_tensor(
                        pyav_video_inter_frames
                    )
                    residuals_and_pts = [
                        [
                            (torch.from_numpy(frame.to_rgb().to_ndarray()), frame.pts)
                            for frame in inter_frames
                        ]
                        for inter_frames in pyav_video_inter_frames
                    ]
                    gop_num_inter_frame = torch.tensor(
                        [len(inter_frames) for inter_frames in pyav_video_inter_frames]
                    )

            if self._has_audio:
                pyav_audio_frames, _ = _pyav_decode_stream(
                    self._container,
                    secs_to_pts(
                        start_secs,
                        self._audio_time_base,
                        self._audio_start_pts,
                        round_mode="ceil",
                    ),
                    secs_to_pts(
                        end_secs,
                        self._audio_time_base,
                        self._audio_start_pts,
                        round_mode="ceil",
                    ),
                    self._container.streams.audio[0],
                    {"audio": 0},
                    perform_seek=self.perform_seek,
                )

                if len(pyav_audio_frames) > 0:
                    audio_and_pts = [
                        (
                            torch.from_numpy(np.mean(frame.to_ndarray(), axis=0)),
                            frame.pts,
                        )
                        for frame in pyav_audio_frames
                    ]

        except Exception as e:
            logger.debug(f"Failed to decode video: {self._video_name}. {e}")

        return {
            "key_frames_and_pts": key_frames_and_pts,
            "motion_vectors_and_pts": motion_vectors_and_pts,
            "residuals_and_pts": residuals_and_pts,
            "gop_num_inter_frame": gop_num_inter_frame,
        }, audio_and_pts


def _transform_mvs_to_tensor(
    gop_inter_frames: list[list[av.video.frame.VideoFrame]],
) -> list[list[tuple[torch.Tensor, int]]]:
    inter_frame_tensor_list: list[list[tuple[torch.Tensor, int]]] = []
    for inter_frames in gop_inter_frames:
        inter_frame_tensors: list[tuple[torch.Tensor, int]] = []
        for frame in inter_frames:
            frame_tensor = torch.zeros(4, frame.height, frame.width, dtype=torch.int32)
            motion_vectors = frame.side_data.get(
                av.sidedata.sidedata.Type.MOTION_VECTORS
            )
            if motion_vectors is not None:
                for mv in motion_vectors:
                    # NOTE: It is important to keep in mind what motion vectors exactly
                    # represent. The raw motion vector for a macroblock of the current frame
                    # saved in a compressed video file points to the region (in a past frame
                    # for list0, and a future frame for list1) where the pixels for the
                    # macroblock should be copied from. This means that while the macroblocks
                    # in the current frame do not overlap by design, the source regions
                    # specified by the motion vectors may. Therefore, we need to represent
                    # motion vectors in terms of the macroblocks in the current frame, rather
                    # than where they came from, which can be more natural and intuitive.
                    # This means that for list0 (forward prediction), motion vectors point to
                    # where the macroblocks of the current frame "came from". On the
                    # other hand, for list1 (backward prediction), motion vectors point to
                    # where the macroblocks of the current frame will "move to."

                    # some motion vectors are 0
                    val_x = mv.src_x - mv.dst_x
                    val_y = mv.src_y - mv.dst_y
                    if val_x == 0 and val_y == 0:
                        continue

                    # (dst_x, dst_y) is the center of the current macroblock
                    start_x = mv.dst_x - mv.w // 2
                    start_y = mv.dst_y - mv.h // 2

                    # list0 is channel 0 and 1, and list1 is channel 2 and 3
                    channel_offset = 0 if mv.source == -1 else 2

                    # fill the x value
                    frame_tensor[
                        channel_offset,
                        start_y : start_y + mv.h,
                        start_x : start_x + mv.w,
                    ] = val_x

                    # fill the y value
                    frame_tensor[
                        channel_offset + 1,
                        start_y : start_y + mv.h,
                        start_x : start_x + mv.w,
                    ] = val_y

            inter_frame_tensors.append((frame_tensor, frame.pts))
        inter_frame_tensor_list.append(inter_frame_tensors)
    return inter_frame_tensor_list


def _pyav_partial_decode_video_stream(
    container: av.container.input.InputContainer, start_pts: int, end_pts: int
) -> tuple[list[av.video.frame.VideoFrame], list[list[av.video.frame.VideoFrame]], int]:
    """
    Decode the video partially with PyAV decoder and return GOPs contained in the
    given Presentation TimeStamps.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the GOPs.
        end_pts (int): the ending Presentation TimeStamp of the decoded GOPs.
    Returns:
        key_frames (list): list of decoded key frames.
        inter_frames (list): list of list of partially decoded inter frames.
        max_pts (int): max Presentation TimeStamp of the video sequence.

        Their lengths equal to the number of GOPs.
    """

    stream = container.streams.video[0]
    stream.codec_context.export_mvs = True

    # NOTE: We need to seek to find key frames since we're decoding GOPs,
    # which start with key frames.
    container.seek(start_pts, stream=stream)

    key_frames: list[av.video.frame.VideoFrame] = []
    inter_frames: list[list[av.video.frame.VideoFrame]] = []
    curr_inter_frames: list[av.video.frame.VideoFrame] = []
    max_pts = 0
    for frame in container.decode(stream):
        max_pts = max(max_pts, frame.pts)
        if frame.pts >= end_pts:
            break
        # pyav, via ffmpeg (libavcodec), returns frames in pts
        if frame.key_frame:
            if len(key_frames) > 0:
                # new key frame (not first), so start collecting
                # a new set of inter frames
                inter_frames.append(curr_inter_frames)
                curr_inter_frames = []
            key_frames.append(frame)
        else:
            curr_inter_frames.append(frame)
    # collect staggler inter frames
    inter_frames.append(curr_inter_frames)

    # sanity check, the length of both lists should be the number of GOPs.
    assert len(key_frames) == len(inter_frames)

    return key_frames, inter_frames, max_pts


def _pyav_decode_stream(
    container: av.container.input.InputContainer,
    start_pts: int,
    end_pts: int,
    stream: av.stream.Stream,
    stream_name: dict,
    perform_seek: bool = True,
) -> tuple[list[av.frame.Frame], float]:
    """
    Decode the video with PyAV decoder in the conventional way of returning
    frames within the given Presentation TimeStamps instead of GOPs.

    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
    Returns:
        result (list): list of decoded frames.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """

    # Seeking in the stream is imprecise. Thus, seek to an earlier pts by a
    # margin pts.
    margin = 1024

    # NOTE:
    # Don't want to seek if iterating through a video due to slow-downs. I
    # believe this is some PyAV bug where seeking after a certain point causes
    # major slow-downs
    if perform_seek:
        seek_offset = max(start_pts - margin, 0)
        container.seek(int(seek_offset), any_frame=False, backward=True, stream=stream)
    frames = {}
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts >= start_pts and frame.pts < end_pts:
            frames[frame.pts] = frame
        elif frame.pts >= end_pts:
            break

    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts
