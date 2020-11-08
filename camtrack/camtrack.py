#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp
import random

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    rodrigues_and_translation_to_view_mat3x4,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    project_points,
    Correspondences,
    compute_reprojection_errors
)


def count_not_none(arr):
    return len(list(filter(lambda x: x is not None, arr)))


def restore_point_cloud(view_mats, corner_storage, last, intrinsic_mat, point_cloud_builder, inl):
    for i, v_mat in enumerate(view_mats):
        if v_mat is None:
            continue
        last_corners = corner_storage[last]
        new_corners = corner_storage[i]
        correspondences = build_correspondences(new_corners, last_corners)
        pts, ids, med = triangulate_correspondences(correspondences, v_mat, view_mats[last],
                                                    intrinsic_mat, TriangulationParameters(5, 1, .1))
        corners_to_add = []
        ids_to_add = []

        for j in ids:
            if j in inl or j not in point_cloud_builder.ids:
                ids_to_add.append(j)
                corners_to_add.append(pts[np.where(ids == j)])

        if len(ids_to_add) > 0:
            point_cloud_builder.add_points(np.array(ids_to_add), np.array(corners_to_add))
    return point_cloud_builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    np.random.seed(1337)
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    corners_1 = corner_storage[known_view_1[0]]
    corners_2 = corner_storage[known_view_2[0]]

    correspondences = build_correspondences(corners_1, corners_2)

    pose_1 = pose_to_view_mat3x4(known_view_1[1])
    pose_2 = pose_to_view_mat3x4(known_view_2[1])

    pts, ids, med = triangulate_correspondences(correspondences, pose_1, pose_2,
                                                intrinsic_mat, TriangulationParameters(5, 1, .1))

    point_cloud_builder = PointCloudBuilder(ids, pts)
    view_mats = [None] * frame_count
    marked_inliers = {}
    last_retr_id = {}
    marked_inliers_p = {}
    marked_inliers[known_view_1[0]] = len(ids)
    marked_inliers[known_view_2[0]] = len(ids)
    view_mats[known_view_1[0]] = pose_1
    view_mats[known_view_2[0]] = pose_2
    last = known_view_2[0]
    level = 0

    inl = []


    while count_not_none(view_mats) < frame_count:
        processed_frames = count_not_none(view_mats) + 1
        print('Processing {0}/{1} frame'.format(processed_frames, frame_count))
        point_cloud_builder = restore_point_cloud(view_mats, corner_storage, last, intrinsic_mat,
                                                  point_cloud_builder, inl)
        print('Points cloud size: {0}'.format(len(point_cloud_builder.points)))
        not_processed = []
        for i, el in enumerate(view_mats):
            if el is None:
                not_processed.append(i)
        # last = np.random.choice(not_processed, 1)[0]
        max_inliers = []
        last = 0
        max_mat = []
        for i in not_processed:
            mat, inliers = get_position(i, point_cloud_builder, corner_storage, intrinsic_mat)
            if len(inliers) > len(max_inliers):
                max_inliers, max_mat, last = inliers, mat, i
        inl = max_inliers
        print('Used {} inliers'.format(len(max_inliers)))
        view_mats[last] = max_mat[:3]
        marked_inliers[last] = len(max_inliers)
        point_cloud_builder = retriangulate_points(last, corner_storage, last_retr_id, level,
                                                   marked_inliers_p, point_cloud_builder, view_mats, intrinsic_mat)
        for i, mat in enumerate(view_mats):
            if mat is None:
                continue
            v_mat, inliers = get_position(i, point_cloud_builder, corner_storage, intrinsic_mat)
            if marked_inliers[i] < len(inliers):
                view_mats[i] = v_mat[:3]
                marked_inliers[i] = len(inliers)
        level += 1

    view_mats = list(filter(lambda x: x is not None, view_mats))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))

    return poses, point_cloud


def get_index_from_intersect(ids_3d, ids_2d, points_1, points_2):
    ids_1 = []
    ids_2 = []
    intersection = snp.intersect(ids_3d.flatten(), ids_2d.flatten(), indices=True)
    for i, point in enumerate(ids_3d):
        if point in intersection[0]:
            ids_1.append(i)
    for i, point in enumerate(ids_2d):
        if point in intersection[0]:
            ids_2.append(i)

    return points_1[ids_1], points_2[ids_2]


def get_position(index, point_cloud_builder, corner_storage, intrinsic_mat):
    pts3d, pts2d = get_index_from_intersect(point_cloud_builder.ids, corner_storage[index].ids,
                                            point_cloud_builder.points,
                                            corner_storage[index].points)
    inliers = cv2.solvePnPRansac(
        objectPoints=pts3d,
        imagePoints=pts2d,
        cameraMatrix=intrinsic_mat,
        distCoeffs=np.array([]),
        iterationsCount=250,
        flags=cv2.SOLVEPNP_EPNP
    )
    _, r_vec, t_vec = cv2.solvePnP(
        objectPoints=pts3d[inliers[3]],
        imagePoints=pts2d[inliers[3]],
        cameraMatrix=intrinsic_mat,
        distCoeffs=np.array([]),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
    return view_mat, inliers[3]


def retriangulate_points(index, corner_storage, last_retr_id, level, marked_inliers_p, point_cloud_builder,
                         view_mats, intrinsic_mat):
    reprojection_error = 0.1
    need = []
    retr_points = []
    retr_ids = []
    for i in corner_storage[index].ids:
        if i[0] not in last_retr_id:
            need.append(i)
            continue
        if level - last_retr_id[i[0]] > 20:
            need.append(i)
    random_indexes = np.random.choice(len(need), 500)
    need = [need[i] for i in random_indexes]
    for i in need:
        points = []
        frames = []
        view_mats_ = []
        for j, frame in enumerate(corner_storage):
            if i in frame.ids and view_mats[j] is not None:
                points.append(frame.points[np.where(frame.ids == i)[0][0]])
                frames.append(j)
                view_mats_.append(view_mats[j])
        if len(points) < 3:
            continue
        if len(points) > 3:
            idxs = np.random.choice(len(points), size=3, replace=False)
            points = np.array(points)[idxs]
            view_mats_ = np.array(view_mats_)[idxs]
        new_points = []
        count = 0
        flag = True
        for _ in range(5):
            rand_ind = np.random.choice(len(points), 2)
            correspondences = Correspondences(np.array([i]), np.array([points[rand_ind[0]]]), np.array([points[rand_ind[1]]]))
            pts, ids, med = triangulate_correspondences(correspondences, view_mats_[rand_ind[0]], view_mats_[rand_ind[1]], intrinsic_mat,
                                                        TriangulationParameters(5, 1, .1))
            if len(pts) > 0:
                err = []
                for mat, point in zip(view_mats_, points):
                    err.append(compute_reprojection_errors(pts, np.array([point]), np.dot(intrinsic_mat, mat)))
                err = np.array(err)
                if flag or count < np.sum(err < reprojection_error):
                    count = np.sum(err < reprojection_error)
                    new_points = pts
                    flag = False
        if flag:
            continue

        if i[0] not in marked_inliers_p or marked_inliers_p[i[0]] < count:
            marked_inliers_p[i[0]] = count
            retr_ids.append(i[0])
            retr_points.append(new_points[0])
            last_retr_id[i[0]] = level
    retr_ids = np.array(retr_ids)
    retr_points = np.array(retr_points)
    if not (len(retr_ids) == 0 or len(retr_points) == 0):
        point_cloud_builder.update_points(retr_ids, retr_points)
    return point_cloud_builder


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
