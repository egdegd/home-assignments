#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from camtrack._corners import FrameCorners, CornerStorage, StorageImpl
from camtrack._corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def find_corners(image_0, image_1, p0, ids, max_corners, last_id):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255.0), np.uint8(image_1 * 255.0), p0, None,
                                           **lk_params)
    p1_rev, st_rev, err_rev = cv2.calcOpticalFlowPyrLK(np.uint8(image_1 * 255.0), np.uint8(image_0 * 255.0), p1,
                                                       None, **lk_params)
    quality = abs(p0 - p1_rev)
    is_good = []
    for el in quality:
        if max(el[0][0], el[0][1]) < 1:
            is_good.append(True)
        else:
            is_good.append(False)
    ids = ids[is_good]
    p0 = p1[is_good]
    if len(p0) < max_corners:
        new_p0 = cv2.goodFeaturesToTrack(image_0, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
        lack_of_points = min(max_corners - len(p0), len(new_p0))
        dist_from_new_points = []
        for point in new_p0.reshape((-1, 2)):
            dist = np.sqrt(np.sum((point - p0) ** 2, axis=2))
            dist_from_new_points.append(dist.min())
        d = np.sort(dist_from_new_points)[-lack_of_points]
        is_good = dist_from_new_points >= d
        new_p0 = new_p0[is_good]
        new_ids = np.array(range(last_id, last_id + lack_of_points))
        last_id += lack_of_points
        ids = np.concatenate([ids, new_ids])
        p0 = np.concatenate([p0, new_p0])
    return p0, ids, last_id


def concatenate_corners(ids, small_ids, p0, small_p0):
    res_ids = np.concatenate([ids, small_ids])
    res_p0 = np.concatenate([p0, small_p0 * 2])
    res_radius = np.concatenate([np.array(np.full(len(p0), 14)), np.array(np.full(len(small_p0), 7))])
    corners = FrameCorners(res_ids, res_p0, res_radius)
    return corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    max_corners = 2000

    p0 = cv2.goodFeaturesToTrack(image_0, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    ids = np.array(range(len(p0)))
    last_id = len(p0)
    compress_image_0 = np.array([el[::2] for el in image_0[::2]])

    max_id = max_corners * len(frame_sequence) + 1
    small_p0 = cv2.goodFeaturesToTrack(compress_image_0, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    small_ids = np.array(range(max_id, max_id + len(small_p0)))
    small_last_id = max_id + len(small_p0)

    corners = concatenate_corners(ids, small_ids, p0, small_p0)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        compress_image_1 = np.array([el[::2] for el in image_1[::2]])
        p0, ids, last_id = find_corners(image_0, image_1, p0, ids, max_corners, last_id)
        small_p0, small_ids, small_last_id = find_corners(compress_image_0, compress_image_1, small_p0, small_ids,
                                                          max_corners, small_last_id)

        corners = concatenate_corners(ids, small_ids, p0, small_p0)
        builder.set_corners_at_frame(frame, corners)

        image_0 = image_1
        compress_image_0 = compress_image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
