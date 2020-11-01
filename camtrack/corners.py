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

from numpy import linalg as LA

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


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


def find_corners(image_0, image_1, p0, ids, radiuses, max_corners, last_id, pyr_lvl, radius):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255.0), np.uint8(image_1 * 255.0), p0, None,
                                           **lk_params)
    p1_rev, st_rev, err_rev = cv2.calcOpticalFlowPyrLK(np.uint8(image_1 * 255.0), np.uint8(image_0 * 255.0), p1,
                                                       None, **lk_params)

    quality = abs(p0 - p1_rev)
    is_good = (LA.norm(quality, axis=2) < 1) & (st == 1)
    is_good = is_good.reshape((-1))
    ids = ids[is_good]
    p0 = p1[is_good]
    radiuses = radiuses[is_good]
    if len(p0) < max_corners:
        lack_of_points = max_corners - len(p0)
        cur_count = int(lack_of_points / (2 - 1 / 2 ** (pyr_lvl - 1)))
        for lvl in range(1, pyr_lvl + 1):
            if cur_count == 0:
                break
            if lvl != 1:
                image_1 = cv2.pyrDown(image_1)
            my_mask = np.full(image_1.shape, 255, dtype=np.uint8)
            p0_ = p0 // (2 ** (lvl - 1))
            for point in p0_:
                x, y = point[0]
                cv2.circle(my_mask, (x, y), radius * (2**lvl), 0, -1)
            new_p0 = cv2.goodFeaturesToTrack(image_1, mask=my_mask, maxCorners=cur_count, qualityLevel=0.01, minDistance=radius // (2**(lvl - 1)))
            if new_p0 is not None:
                new_p0 = new_p0 * (2 ** (lvl - 1))
                new_ids = np.array(range(last_id, last_id + len(new_p0)))
                last_id += len(new_p0)
                ids = np.concatenate([ids, new_ids])
                p0 = np.concatenate([p0, new_p0[:len(new_p0)]])
                radiuses = np.concatenate([radiuses, np.array(np.full(len(new_p0), radius * (2**(lvl - 1))))])
            cur_count //= 2
    return p0, ids, radiuses, last_id


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    max_corners = 2500
    pyr_lvl = 3
    radius = 4
    cur_count = int(max_corners / (2 - 1 / 2 ** (pyr_lvl - 1)))
    p0 = cv2.goodFeaturesToTrack(image_0, maxCorners=cur_count, qualityLevel=0.01, minDistance=7)
    ids = np.array(range(len(p0)))
    radiuses = np.array(np.full(len(p0), radius))
    last_id = len(p0)

    compress_image_0 = image_0

    for lvl in range(2, pyr_lvl + 1):
        cur_count //= 2
        compress_image_0 = cv2.pyrDown(compress_image_0)
        compress_p0 = cv2.goodFeaturesToTrack(compress_image_0, maxCorners=cur_count, qualityLevel=0.01, minDistance=7)
        compress_ids = np.array(range(last_id, last_id + len(compress_p0)))
        compress_radiuses = np.array(np.full(len(compress_p0), radius *
                                             (2**(lvl - 1))))
        last_id = last_id + len(compress_p0)

        ids = np.concatenate([ids, compress_ids])
        p0 = np.concatenate([p0, compress_p0 * 2**(lvl - 1)])
        radiuses = np.concatenate([radiuses, compress_radiuses])
    corners = FrameCorners(ids, p0, radiuses)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        p0, ids, radiuses, last_id = find_corners(image_0, image_1, p0, ids, radiuses, max_corners, last_id, pyr_lvl, radius)
        corners = FrameCorners(ids, p0, radiuses)
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1



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
