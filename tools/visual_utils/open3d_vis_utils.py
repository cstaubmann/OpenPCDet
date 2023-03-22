"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, pc_plane=None,
                point_colors=None, point_size=1.0, bg_color=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = point_size
    if bg_color is None:
        vis.get_render_option().background_color = np.zeros(3)
    else:
        vis.get_render_option().background_color = bg_color

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        p_color = np.tile(point_colors, points.shape[0]).reshape(points.shape[0], 3)
        pts.colors = open3d.utility.Vector3dVector(p_color)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    if pc_plane is not None:
        vis = draw_plane(vis, pc_plane, (0.3, 0, 0))

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_plane(vis, pc_plane, color=(0.5, 0, 0)):
    [pcd, plane_model, inliers] = pc_plane

    [a, b, c, d] = plane_model
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0.7, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    vis.add_geometry(inlier_cloud)
    # open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # draw plane using triangle meshes
    plane_triangle_1 = open3d.geometry.TriangleMesh()
    plane_triangle_2 = open3d.geometry.TriangleMesh()
    l_min = -3.0
    l_max = +3.0
    vertices_1 = np.array([[l_min, l_min, -d],
                           [l_max, l_min, -d],
                           [l_max, l_max, -d]])
    triangle_1 = np.array([[0, 1, 2]]).astype(np.int32)
    vertices_2 = np.array([[l_min, l_min, -d],
                           [l_min, l_max, -d],
                           [l_max, l_max, -d]])
    triangle_2 = np.array([[0, 2, 1]]).astype(np.int32)
    plane_triangle_1.vertices = open3d.utility.Vector3dVector(vertices_1)
    plane_triangle_1.triangles = open3d.utility.Vector3iVector(triangle_1)
    plane_triangle_1.paint_uniform_color(color)
    plane_triangle_2.vertices = open3d.utility.Vector3dVector(vertices_2)
    plane_triangle_2.triangles = open3d.utility.Vector3iVector(triangle_2)
    plane_triangle_2.paint_uniform_color(color)

    vis.add_geometry(plane_triangle_1)
    vis.add_geometry(plane_triangle_2)

    return vis
