import pickle
import numpy as np
from pathlib import Path

gt_boxes_name = {
    'ONCEDataset': 'boxes_3d',
    'CadcDataset': 'gt_boxes_lidar',
}


def extract_split_info(split_info, dataset, class_names, split, save_path):

    num_annotations = 0
    num_scenes_with_annos = 0
    num_scenes_without_annos = 0
    num_class_dict = {key: 0 for key in class_names}
    lidar_point_class_dict = {key: [] for key in class_names}
    bbox_size_class_dict = {key: [] for key in class_names}

    for info_entry in split_info:
        if 'annos' not in info_entry:
            num_scenes_without_annos += 1
            continue
        num_scenes_with_annos += 1
        cur_names = info_entry['annos']['name']
        num_annotations += len(cur_names)
        # Count class instances
        for k, v in num_class_dict.items():
            class_count_in_entry = np.count_nonzero(cur_names == k)
            num_class_dict[k] = v + class_count_in_entry
        # LiDAR points per class + bbox sizes per class
        gt_boxes = gt_boxes_name[dataset] if dataset in gt_boxes_name else 'boxes_3d'
        cur_boxes3d = info_entry['annos'][gt_boxes]
        cur_points_gt = info_entry['annos']['num_points_in_gt']
        assert len(cur_names) == len(cur_boxes3d) == len(cur_points_gt)
        for i in range(len(cur_names)):
            cur_class_name = str(cur_names[i])
            cur_box_3d_lwh = cur_boxes3d[i][3:6]
            cur_num_gt_points = cur_points_gt[i]
            lidar_point_class_dict[cur_class_name].append(cur_num_gt_points)
            bbox_size_class_dict[cur_class_name].append(cur_box_3d_lwh)

    # ---------- Save extracted info as .pkl file ----------

    output_dir = save_path / 'split_infos'
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = 'split_info_once_%s.pkl' % split
    filename = output_dir / Path(filename)

    with open(filename, 'wb') as save_file:
        pickle.dump(num_annotations, save_file)
        pickle.dump(num_scenes_with_annos, save_file)
        pickle.dump(num_scenes_without_annos, save_file)
        pickle.dump(num_class_dict, save_file)
        pickle.dump(lidar_point_class_dict, save_file)
        pickle.dump(bbox_size_class_dict, save_file)

    print('ONCE %s split info is saved to %s' % (split, filename))

    filename = 'split_info_once_%s.txt' % split
    filename = output_dir / Path(filename)

    # ---------- Save extracted info as readable .txt file ----------

    with open(filename, 'w') as save_file:
        save_file.write(f"Split: {split}\n")

        save_file.write('\tAverage data points in gt (per class):\n')
        for name, points_array in lidar_point_class_dict.items():
            save_file.write(f'\t\t{name}:\t{np.mean(points_array):.2f}\n')

        save_file.write('Average 3d bounding box sizes (per class) in "lwh" format:\n')
        for name, bbox_array in bbox_size_class_dict.items():
            np_array = np.array(bbox_array)
            avg_length = np.mean(np_array, axis=0)
            save_file.write(f'\t\t{name}:\t [{avg_length[0]:.2f}, {avg_length[1]:.2f}, {avg_length[2]:.2f}]\n')

    print('ONCE %s split info is saved to %s' % (split, filename))
