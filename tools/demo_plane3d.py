import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--dont_draw_scenes', dest='draw_scenes', action='store_false', default=True,
                        help='dont draw scenes from demo dataset')
    parser.add_argument('--draw_gt_boxes', dest='draw_gt_boxes', action='store_true', default=False,
                        help='draw groundtruth boxes in scene')
    parser.add_argument('--fit_plane', dest='fit_plane', action='store_true', default=False,
                        help='draw best fitting plane for points pointcloud')
    parser.add_argument('--log_file', type=str, default='', help='filename (without ext) to save logger output to disk')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    log_dir = cfg.ROOT_DIR / 'output' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ('%s.txt' % args.log_file.replace('.txt', '')) if args.log_file else None
    logger = common_utils.create_logger(log_file=log_file)
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger
    # )
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    from pcdet.datasets import ONCEDataset, CadcDataset
    __all__ = {
        'DatasetTemplate': DatasetTemplate,
        'ONCEDataset': ONCEDataset,
        'CadcDataset': CadcDataset
    }

    demo_dataset = __all__[cfg.DATA_CONFIG.DATASET](
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.data_path),
        training=False,
        logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        # ? numpy array for plane fitting
        np_plane_models = np.array([])

        for idx, data_dict in enumerate(demo_dataset):
            if cfg.DATA_CONFIG.DATASET == 'CadcDataset':
                gt_boxes = data_dict['gt_boxes'] if args.draw_gt_boxes else None
                sample_idx = data_dict['sample_idx']
                sample_date = sample_idx[0]
                sample_sequence = sample_idx[1]
                sample_timestamp = sample_idx[2]
                logger.info(f'Dataset sample index:\t{idx}:'
                            f'\t{sample_date} \t{sample_sequence} \t{sample_timestamp}')
            elif cfg.DATA_CONFIG.DATASET == 'ONCEDataset':
                gt_boxes = data_dict['gt_boxes'] if args.draw_gt_boxes else None
                sample_frameid = data_dict['frame_id']
                logger.info(f'Dataset sample index:\t{idx}:\t{sample_frameid}')
            else:
                logger.info(f'Dataset sample index: \t{idx}')

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pc_plane = None
            if args.fit_plane:
                # TODO: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Plane-segmentation
                #       http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html#from-numpy-to-open3d-pointcloud
                # ? segmententation of geometric primitives from point clouds using RANSAC
                # ? find the plane with the largest support in the point cloud
                pcd = open3d.geometry.PointCloud()
                np_points = data_dict['points'][:, 1:-1].cpu().numpy()  # np-array (N, 3): cut off reflection intensity
                pcd.points = open3d.utility.Vector3dVector(np_points)
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                                         ransac_n=3,
                                                         num_iterations=1000)
                [a, b, c, d] = plane_model
                logger.info(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
                # ? Filter out planes along different axes:
                if +0.05 >= a >= -0.05 and +0.05 >= b >= -0.05 and 1.0 >= c >= 0.95:
                    np_plane_models = np.append(np_plane_models, plane_model)
                    pc_plane = [pcd, plane_model, inliers]

            if not args.draw_scenes:
                continue
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'], gt_boxes=gt_boxes,
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                pc_plane=pc_plane
            )
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    if args.fit_plane:
        np_plane_models = np_plane_models.reshape((-1, 4))
        np_plane_d = np_plane_models[:, -1:]
        plane_d_min = np.min(np_plane_d)
        plane_d_max = np.max(np_plane_d)
        plane_d_mean = np.mean(np_plane_d)
        plane_d_median = np.median(np_plane_d)
        logger.info(f"Scenes with relevant planes: ({len(np_plane_d)} / {len(demo_dataset)})")
        logger.info(f"Plane D min:    \t {plane_d_min}")
        logger.info(f"Plane D max:    \t {plane_d_max}")
        logger.info(f"Plane D mean:   \t {plane_d_mean}")
        logger.info(f"Plane D median: \t {plane_d_median}")

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
