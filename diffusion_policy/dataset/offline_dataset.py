from typing import Dict
from abc import abstractmethod
import torch
import numpy as np
import copy
from diffusion_policy.utils.pytorch_util import dict_apply
from diffusion_policy.utils.replay_buffer import ReplayBuffer
from diffusion_policy.utils.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.utils.normalizer import LinearNormalizer


class classproperty(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


class BaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseDataset":
        # return an empty dataset by default
        return BaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    # ---------------- Normalize w.r.t. local frame ----------------
    @staticmethod
    def state_normalize():
        pass

    @staticmethod
    def state_unnormalize():
        pass


class OfflineDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        n_past_steps=3,
        pad_before=0,
        pad_after=0,
        obs_key="keypoint",
        state_key="state",
        action_key="action",
        seed=42,
        val_ratio=0.0,
        **kwargs
    ):

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[
                "act",
                "body_ang_vel",
                "body_lin_vel",
                "body_pos",
                "body_rot",
                "joint_pos",
                "joint_vel",
                "root_pos",
                "root_rot",
            ],
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_past_steps = n_past_steps

        self.obs_reflect_op, self.action_reflect_op = self.get_reflection_ops()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {
            "obs": sample[self.state_key],  # T, D_o
            "action": sample[self.action_key],  # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def collate_fn(self, batch):

        obs_stack = torch.stack([item["obs"] for item in batch])
        act_stack = torch.stack([item["action"] for item in batch])

        return {"obs": obs_stack, "action": act_stack}

    @classproperty
    def body_names(self):
        return BODY_NAMES

    @classproperty
    def joint_names(cls):
        return JOINT_NAMES

    def get_normalizer(self, mode="limits", **kwargs):
        data = self._sample_to_data(self.replay_buffer)

        where_non_padded = np.where(
            np.logical_and(
                self.sampler.indices[:, 3] == self.horizon,
                self.sampler.indices[:, 2] == 0,
            )
        )
        non_padded_indices = self.sampler.indices[where_non_padded]
        buffer_start_idx = non_padded_indices[:, 0]

        buffer_start_idx = buffer_start_idx[
            np.random.choice(
                len(buffer_start_idx), int(len(buffer_start_idx) * 0.1), replace=False
            )
        ]

        indices = buffer_start_idx[:, None] + np.arange(self.horizon)

        keys = data.keys()
        tensor_data = {key: torch.tensor(data[key][indices]) for key in keys}
        batch = [
            {key: tensor_data[key][i] for key in keys} for i in range(len(indices))
        ]

        # Apply Collate Function
        normalized_batch = self.collate_fn(batch, normalize=True)
        data["obs"] = normalized_batch["obs"].clone()
        data["action"] = normalized_batch["action"].clone()

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    @staticmethod
    @abstractmethod
    def state_normalize(
        root_pos_frame,
        root_rot_frame,
        body_pos,
        body_rot,
        body_lin_vel,
        nominal_frame_idx,
    ):
        pass

    @staticmethod
    @abstractmethod
    def state_unnormalize(state):
        pass

    @classmethod
    @abstractmethod
    def get_reflection_ops(cls):
        pass

    def symmetric_augment(self, obs, action):
        obs = torch.cat([obs, obs @ self.obs_reflect_op], dim=0)
        action = torch.cat([action, action @ self.action_reflect_op], dim=0)
        return obs, action
