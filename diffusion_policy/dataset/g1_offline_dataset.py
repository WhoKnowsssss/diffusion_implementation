from typing import Dict, List, Optional
import torch
import numpy as np
from diffusion_policy.dataset.offline_dataset import OfflineDataset, classproperty, BODY_NAMES
from diffusion_policy.utils.traj_utils import (
    quat_from_euler_xyz, get_euler_xyz, quat_mul, quat_rotate,
    box_minus, quat_rotate_inverse, box_plus, get_yaw_quat,
)
from diffusion_policy.utils.symm_utils import get_reflect_reps, get_reflect_op


class G1DatasetBase(OfflineDataset):
    """Base class with common functionality for G1 datasets"""
    
    def __init__(self, symm_aug=True, **kwargs):
        super().__init__(**kwargs)
        self.symm_aug = symm_aug

    @staticmethod
    def ee_idxs():
        """Override in subclasses to provide end-effector indices"""
        raise NotImplementedError

    def _sample_to_data(self, sample):
        return {
            "root_pos": sample["root_pos"],
            "root_rot": sample["root_rot"],
            "body_rot": sample["body_rot"],
            "body_pos": sample["body_pos"],
            "body_lin_vel": sample["body_lin_vel"],
            "body_ang_vel": sample["body_ang_vel"],
            "joint_pos": sample["joint_pos"],
            "action": sample["act"],
        }

    @classmethod
    def get_reflection_ops(cls):
        """Get reflection operations for symmetry augmentation"""
        ee_idxs = cls.ee_idxs()
        
        Q, Rd, Rd_pseudo, Q_Rd, Q_Rd_pseudo, num_bodies = get_reflect_reps(cls.body_names, cls.joint_names)

        Q_Rd_pseudo = Q_Rd_pseudo.view(num_bodies, 3, num_bodies, 3)
        Q_Rd_pseudo_ee = Q_Rd_pseudo[ee_idxs][:, :, ee_idxs]
        Q_Rd_pseudo_ee = Q_Rd_pseudo_ee.view(len(ee_idxs) * 3, len(ee_idxs) * 3)
        
        obs_reflect_reps = [Q_Rd] * 2 + [Rd] + [Rd_pseudo] + [Rd] + [Rd_pseudo]
        action_reflect_reps = [Q]

        obs_reflect_op = get_reflect_op(obs_reflect_reps).to(torch.float64)
        action_reflect_op = get_reflect_op(action_reflect_reps).to(torch.float64)
        return obs_reflect_op, action_reflect_op

    @staticmethod
    def _compute_yaw_frame(root_rot_frame):
        """Get character heading for normalization"""
        B, H = root_rot_frame.shape[:2]
        first_quaternion = root_rot_frame.reshape(-1, 4)
        roll, pitch, yaw = get_euler_xyz(first_quaternion)
        return quat_from_euler_xyz(roll * 0, pitch * 0, yaw).reshape(B, H, 4)
    
    @staticmethod
    def _normalize_positions( body_pos, root_pos_frame, yaw_quat, nominal_frame_idx):
        """Normalize body positions relative to root and yaw frame"""
        B, H, J = body_pos.shape[:3]
        root_pos_init_frame = root_pos_frame[:, nominal_frame_idx]
        
        # Remove root translation
        body_pos[:, :, :, :2] -= root_pos_frame[:, :, None, :2]
        
        # Rotate to yaw frame
        body_pos_local = quat_rotate_inverse(
            yaw_quat[:, :, None, :].repeat(1, 1, J, 1).reshape(-1, 4),
            body_pos.reshape(-1, 3),
        ).reshape(B, H, J, -1)
        
        return body_pos_local
    
    @staticmethod
    def _normalize_rotations(body_rot, yaw_quat):
        """Normalize body rotations relative to yaw frame"""
        B, H, J = body_rot.shape[:3]
        body_rot = body_rot.reshape(-1, 4)
        body_rot_local = box_minus(
            body_rot, yaw_quat[:, :, None, :].repeat(1, 1, J, 1).reshape(-1, 4)
        ).reshape(B, H, J, -1)
        return body_rot_local
    
    @staticmethod
    def _normalize_velocities( body_lin_vel, body_ang_vel, yaw_quat, nominal_frame_idx):
        """Normalize linear and angular velocities"""
        if body_lin_vel is None or body_ang_vel is None:
            return None, None, None, None
            
        try:
            B, H, J = body_lin_vel.shape[:3]
            
            # Linear velocities
            root_lin_vel_local = body_lin_vel[:, :, 0, :].clone()
            body_lin_vel_local = body_lin_vel.clone() - root_lin_vel_local[:, :, None, :]
            
            body_lin_vel_local = quat_rotate_inverse(
                yaw_quat[:, :, None, :].repeat(1, 1, J, 1).reshape(-1, 4),
                body_lin_vel_local.reshape(-1, 3),
            ).reshape(B, H, J, -1)
            
            root_lin_vel_local = quat_rotate_inverse(
                yaw_quat[:, nominal_frame_idx:nominal_frame_idx + 1, :]
                .repeat(1, H, 1).reshape(-1, 4),
                root_lin_vel_local.reshape(-1, 3),
            ).reshape(B, H, -1)
            
            # Angular velocities
            root_ang_vel_local = body_ang_vel[:, :, 0, :].clone()
            root_ang_vel_local = quat_rotate_inverse(
                yaw_quat[:, nominal_frame_idx:nominal_frame_idx + 1, :]
                .repeat(1, H, 1).reshape(-1, 4),
                root_ang_vel_local.reshape(-1, 3),
            ).reshape(B, H, -1)
            
            return body_lin_vel_local, root_lin_vel_local, root_ang_vel_local
            
        except Exception:
            return None, None, None
    
    @staticmethod
    def _normalize_root_pose(root_pos_frame, root_rot_frame, yaw_quat, nominal_frame_idx):
        """Normalize root position and rotation"""
        B, H = root_pos_frame.shape[:2]
        root_pos_init_frame = root_pos_frame[:, nominal_frame_idx]
        
        # Root position
        root_pos_local = root_pos_frame.clone()
        root_pos_local[:, :, :2] = root_pos_frame[:, :, :2] - root_pos_init_frame[:, None, :2]
        root_pos_local = quat_rotate_inverse(yaw_quat[:, nominal_frame_idx:nominal_frame_idx + 1, :].repeat(1, H, 1).reshape(-1, 4), root_pos_local.reshape(-1, 3),).reshape(B, H, -1)
        
        # Root rotation
        root_rot_frame = root_rot_frame.reshape(-1, 4)
        root_rot_local = box_minus(root_rot_frame,yaw_quat[:, nominal_frame_idx:nominal_frame_idx + 1, :].repeat(1, H, 1).reshape(-1, 4),).reshape(B, H, -1)
        
        return root_pos_local, root_rot_local

    @staticmethod
    def state_unnormalize(state, global_root=None, return_rot=False):
        """Unnormalize state from local coordinates back to global"""
        body_pos_local = state[:, :, 0 : 0 + 90]
        root_pos_frame = state[:, :, 180:183]
        root_rot_frame = state[:, :, 183:186]
        B, H = state.shape[:2]
        body_pos_local = body_pos_local.view(B, H, -1, 3)
        J = body_pos_local.shape[2]
        body_pos = body_pos_local.clone()

        if global_root is not None:
            root_pos_global = global_root[:, :, 0:3].to(body_pos.device)
            root_rot_global = global_root[:, :, 3:7].to(body_pos.device)
            root_rot_global = get_yaw_quat(root_rot_global)
        else:
            root_rot_global = torch.zeros(
                (*body_pos_local.shape[:2], 4), device=body_pos_local.device
            )
            root_rot_global[..., 0] = 1

        root_rot_frame = box_plus(
            root_rot_global.flatten(0, 1), -root_rot_frame.flatten(0, 1)
        ).view(B, H, 4)

        roll, pitch, yaw = get_euler_xyz(root_rot_frame.view(-1, 4))
        yaw_quat = quat_from_euler_xyz(roll * 0, pitch * 0, yaw).reshape(B, H, 4)

        body_pos = quat_rotate(
            yaw_quat[:, :, None, :].repeat(1, 1, J, 1).flatten(0, 2),
            body_pos.flatten(0, 2),
        ).view(B, H, J, 3)
        body_pos[..., :2] += quat_rotate(
            root_rot_global[:, :, None, :].repeat(1, 1, J, 1).flatten(0, 2),
            root_pos_frame.flatten(0, 2),
        ).view(B, H, J, 3)[..., :2]
        body_pos[..., :2] += root_pos_global[:, :, None, :2]

        return body_pos

    def collate_fn(self, batch, normalize=False):
        """Collate batch data"""
        B = len(batch)
        
        # Stack batch data
        root_pos_frame = torch.stack([item["root_pos"] for item in batch])
        root_rot_frame = torch.stack([item["root_rot"] for item in batch])
        joint_pos = torch.stack([item["joint_pos"] for item in batch])
        action = torch.stack([item["action"] for item in batch])
        
        B, H = root_rot_frame.shape[:2]
        body_pos = torch.stack([item["body_pos"] for item in batch]).view(B, H, -1, 3)
        body_rot = torch.stack([item["body_rot"] for item in batch]).view(B, H, -1, 4)
        body_lin_vel = torch.stack([item["body_lin_vel"] for item in batch]).view(B, H, -1, 3)
        body_ang_vel = torch.stack([item["body_ang_vel"] for item in batch]).view(B, H, -1, 3)

        # Normalize state  
        obs_stack = self.__class__.state_normalize(root_pos_frame, root_rot_frame, body_pos, body_rot, body_lin_vel, body_ang_vel, self.n_past_steps - 1, self.__class__.ee_idxs(), joint_pos)

        # Apply symmetry augmentation if configured (doubles batchsize)
        if self.symm_aug:
            obs_stack, action = self.symmetric_augment(obs_stack, action)

        return {"obs": obs_stack, "action": action}


class G1_Dataset(G1DatasetBase):
    """Standard G1 dataset"""
    
    @staticmethod
    def ee_idxs():
        return np.array([17, 18])
    
    @staticmethod
    def state_normalize(
        root_pos_frame,
        root_rot_frame,
        body_pos,
        body_rot,
        body_lin_vel,
        body_ang_vel,
        nominal_frame_idx,
        ee_idxs,
        joint_pos,
        return_raw=False,
    ):
        """Normalize state components to local coordinate frame"""
        B, H, J = body_pos.shape[:3]
        
        # Compute yaw frame for normalization
        yaw_quat = G1DatasetBase._compute_yaw_frame(root_rot_frame)
        
        # Normalize all components
        body_pos_local = G1DatasetBase._normalize_positions(body_pos, root_pos_frame, yaw_quat, nominal_frame_idx)
        root_pos_local, root_rot_local = G1DatasetBase._normalize_root_pose(root_pos_frame, root_rot_frame, yaw_quat, nominal_frame_idx)
        
        # Velocities (with error handling)
        body_lin_vel_local, root_lin_vel_local, root_ang_vel_local = G1DatasetBase._normalize_velocities(
            body_lin_vel, body_ang_vel, yaw_quat, nominal_frame_idx
        )

        obs_stack = torch.cat(
            (
                body_pos_local.reshape(B, H, -1),
                body_lin_vel_local.reshape(B, H, -1),
                root_pos_local.reshape(B, H, -1),
                root_rot_local.reshape(B, H, -1),
                root_lin_vel_local.reshape(B, H, -1),
                root_ang_vel_local.reshape(B, H, -1),
            ),
            dim=-1,
        ).reshape(B, H, -1)

        return obs_stack


class G1_Dataset_EE(G1DatasetBase):
    """G1 dataset with end-effector rotations"""
    
    @staticmethod
    def ee_idxs():
        return np.array([17, 18])
    
    @staticmethod
    def state_normalize(
        root_pos_frame,
        root_rot_frame,
        body_pos,
        body_rot,
        body_lin_vel,
        body_ang_vel,
        nominal_frame_idx,
        ee_idxs,
        joint_pos,
        return_raw=False,
    ):
        """Normalize state components to local coordinate frame"""
        B, H, J = body_pos.shape[:3]
        
        # Compute yaw frame for normalization
        yaw_quat = G1DatasetBase._compute_yaw_frame(root_rot_frame)
        
        # Normalize all components
        body_pos_local = G1DatasetBase._normalize_positions(body_pos, root_pos_frame, yaw_quat, nominal_frame_idx)
        body_rot_local = G1DatasetBase._normalize_rotations(body_rot, yaw_quat)
        root_pos_local, root_rot_local = G1DatasetBase._normalize_root_pose(root_pos_frame, root_rot_frame, yaw_quat, nominal_frame_idx)
        
        # Velocities (with error handling)
        body_lin_vel_local, root_lin_vel_local, root_ang_vel_local = G1DatasetBase._normalize_velocities(
            body_lin_vel, body_ang_vel, yaw_quat, nominal_frame_idx
        )

        # End-effector rotations
        ee_rot_local = body_rot_local[:, :, ee_idxs, :]
        
        obs_stack = torch.cat(
            (
                body_pos_local.reshape(B, H, -1),
                body_lin_vel_local.reshape(B, H, -1),
                root_pos_local.reshape(B, H, -1),
                root_rot_local.reshape(B, H, -1),
                root_lin_vel_local.reshape(B, H, -1),
                root_ang_vel_local.reshape(B, H, -1),
                ee_rot_local.reshape(B, H, -1),
            ),
            dim=-1,
        ).reshape(B, H, -1)

        return obs_stack


class G1_Dataset_limited(G1DatasetBase):
    """G1 dataset with limited body set"""
    
    LIMITED_BODY_NAMES = [
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link",
        "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
        "right_wrist_yaw_link",
    ]
    
    @staticmethod
    def ee_idxs():
        return np.array([18, 19])
    
    @classmethod
    def state_normalize(
        cls,
        root_pos_frame,
        root_rot_frame,
        body_pos,
        body_rot,
        body_lin_vel,
        body_ang_vel,
        nominal_frame_idx,
        ee_idxs,
        joint_pos,
        return_raw=False,
    ):
        """Normalize state components to local coordinate frame"""
        B, H, J = body_pos.shape[:3]
        
        # Compute yaw frame for normalization
        yaw_quat = G1DatasetBase._compute_yaw_frame(root_rot_frame, nominal_frame_idx)
        
        # Normalize all components  
        body_pos_local = G1DatasetBase._normalize_positions(body_pos, root_pos_frame, yaw_quat, nominal_frame_idx)
        root_pos_local, root_rot_local = G1DatasetBase._normalize_root_pose(root_pos_frame, root_rot_frame, yaw_quat, nominal_frame_idx)
        
        # Velocities (with error handling)
        body_lin_vel_local, root_lin_vel_local, root_ang_vel_local = G1DatasetBase._normalize_velocities(
            body_lin_vel, body_ang_vel, yaw_quat, nominal_frame_idx
        )

        # Filter to only the bodies we want
        body_idxs = [BODY_NAMES.index(name) for name in cls.LIMITED_BODY_NAMES]
        
        obs_stack = torch.cat(
            (
                body_pos_local[:, :, body_idxs].reshape(B, H, -1),
                body_lin_vel_local[:, :, body_idxs].reshape(B, H, -1),
                root_pos_local.reshape(B, H, -1),
                root_rot_local.reshape(B, H, -1),
                root_lin_vel_local.reshape(B, H, -1),
                root_ang_vel_local.reshape(B, H, -1),
            ),
            dim=-1,
        ).reshape(B, H, -1)

        return obs_stack
    
    @classproperty
    def body_names(self):
        return self.LIMITED_BODY_NAMES