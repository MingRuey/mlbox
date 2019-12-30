import os
import sys
import pathlib

import pytest
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip

from Database.formats import VIDEOFMT


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestVideoFormat:

    sample_videos = list(
        pathlib.Path(__file__).parent.joinpath("sample_files").glob("*.mp4")
    )
    label_map = {
        file.stem: True for file in sample_videos
    }

    def test_to_tf_sequence_example(self):
        """the converted tf sequence example should contain the video data"""
        assert self.sample_videos
        fmt = VIDEOFMT(id_label_map=self.label_map)

        for file in self.sample_videos:
            sequence_example = fmt.to_tfexample(str(file))
            sequence_example = sequence_example.SerializeToString()

            parser = fmt.get_parser()
            video, audio, label = parser(sequence_example)

            video = video.numpy()
            audio = audio.numpy()
            label = label.numpy()

            gt_video = VideoFileClip(str(file), audio=True)
            gt_audio = gt_video.audio

            gt_video = next(
                frame for i, frame in
                enumerate(gt_video.iter_frames()) if i == 10
            )

            gt_audio = next(
                sound for i, sound in
                enumerate(gt_audio.iter_frames()) if i == 10
            )

            assert label == np.array([1])
            assert np.all(video[10, ...] == gt_video)
            assert np.all(audio[10, ...] == gt_audio)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
