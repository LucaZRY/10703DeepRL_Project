"""
Data Persistence and Loading Utilities for Expert Data

This module provides utilities for saving, loading, and managing expert demonstration
datasets with support for multiple formats and efficient data handling.

Usage:
    from data_utils import DataManager

    # Save dataset
    dm = DataManager()
    dm.save_dataset(dataset, "expert_data.pkl", format="pickle")

    # Load dataset
    dataset = dm.load_dataset("expert_data.pkl")

    # Batch loading
    loader = dm.create_dataloader(dataset, batch_size=64)
"""

import os
import pickle
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import gzip
import shutil
from pathlib import Path
import warnings


class ExpertDataset(Dataset):
    """
    PyTorch Dataset wrapper for expert demonstration data.
    """

    def __init__(self, observations: List[np.ndarray], actions: List[np.ndarray],
                 rewards: Optional[List[float]] = None, metadata: Optional[Dict] = None):
        """
        Initialize dataset.

        Args:
            observations: List of observation arrays
            actions: List of action arrays
            rewards: Optional list of rewards
            metadata: Optional metadata dictionary
        """
        self.observations = observations
        self.actions = actions
        self.rewards = rewards or [0.0] * len(observations)
        self.metadata = metadata or {}

        # Validate data consistency
        assert len(self.observations) == len(self.actions), \
            f"Observation and action counts don't match: {len(self.observations)} vs {len(self.actions)}"

        if len(self.rewards) != len(self.observations):
            warnings.warn("Reward count doesn't match observation count")
            self.rewards = [0.0] * len(self.observations)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Get item at index.

        Args:
            idx: Index to retrieve

        Returns:
            Tuple of (observation, action, reward)
        """
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        reward = float(self.rewards[idx])

        return obs, action, reward

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if len(self.observations) == 0:
            return {}

        # Calculate observation statistics
        obs_sample = self.observations[0]
        obs_shape = obs_sample.shape

        # Calculate action statistics
        actions_array = np.array(self.actions)
        action_stats = {
            'mean': np.mean(actions_array, axis=0).tolist(),
            'std': np.std(actions_array, axis=0).tolist(),
            'min': np.min(actions_array, axis=0).tolist(),
            'max': np.max(actions_array, axis=0).tolist()
        }

        # Calculate reward statistics
        reward_stats = {
            'mean': float(np.mean(self.rewards)),
            'std': float(np.std(self.rewards)),
            'min': float(np.min(self.rewards)),
            'max': float(np.max(self.rewards))
        }

        return {
            'size': len(self),
            'observation_shape': obs_shape,
            'action_dim': len(self.actions[0]),
            'action_stats': action_stats,
            'reward_stats': reward_stats,
            'metadata': self.metadata
        }


class DataManager:
    """
    Comprehensive data management system for expert datasets.
    """

    def __init__(self, base_dir: str = "data"):
        """
        Initialize data manager.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Supported formats
        self.supported_formats = ['pickle', 'hdf5', 'npz', 'json']

    def save_dataset(self, dataset: Union['EnhancedImitationDataset', ExpertDataset],
                    filename: str, format: str = "pickle",
                    compress: bool = True, include_metadata: bool = True) -> str:
        """
        Save dataset in specified format.

        Args:
            dataset: Dataset to save
            filename: Output filename
            format: Format to use ('pickle', 'hdf5', 'npz', 'json')
            compress: Whether to compress the file
            include_metadata: Whether to include metadata

        Returns:
            Path to saved file
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")

        filepath = self.base_dir / filename
        timestamp = datetime.now().isoformat()

        # Prepare data for saving
        if hasattr(dataset, 'observations'):
            # EnhancedImitationDataset format
            save_data = {
                'observations': dataset.observations,
                'actions': dataset.actions,
                'rewards': dataset.rewards,
                'next_observations': getattr(dataset, 'next_observations', None),
                'dones': getattr(dataset, 'dones', None),
                'episode_ids': getattr(dataset, 'episode_ids', None),
                'step_ids': getattr(dataset, 'step_ids', None),
                'collection_methods': getattr(dataset, 'collection_methods', None),
                'metadata': getattr(dataset, 'metadata', None)
            }
        else:
            # ExpertDataset format
            save_data = {
                'observations': dataset.observations,
                'actions': dataset.actions,
                'rewards': dataset.rewards,
                'metadata': dataset.metadata
            }

        if include_metadata:
            save_data['save_metadata'] = {
                'timestamp': timestamp,
                'format': format,
                'compressed': compress,
                'dataset_type': type(dataset).__name__,
                'dataset_size': len(dataset)
            }

        # Save in specified format
        if format == "pickle":
            filepath = filepath.with_suffix('.pkl')
            self._save_pickle(save_data, filepath, compress)

        elif format == "hdf5":
            filepath = filepath.with_suffix('.h5')
            self._save_hdf5(save_data, filepath, compress)

        elif format == "npz":
            filepath = filepath.with_suffix('.npz')
            self._save_npz(save_data, filepath, compress)

        elif format == "json":
            filepath = filepath.with_suffix('.json')
            self._save_json(save_data, filepath, compress)

        print(f"[DataManager] Dataset saved to {filepath}")
        print(f"[DataManager] Format: {format}, Size: {len(dataset)}, Compressed: {compress}")

        return str(filepath)

    def load_dataset(self, filepath: str, format: Optional[str] = None) -> Union['EnhancedImitationDataset', ExpertDataset]:
        """
        Load dataset from file.

        Args:
            filepath: Path to dataset file
            format: Force specific format (auto-detected if None)

        Returns:
            Loaded dataset
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(filepath)

        print(f"[DataManager] Loading dataset from {filepath} (format: {format})")

        # Load data based on format
        if format == "pickle":
            data = self._load_pickle(filepath)
        elif format == "hdf5":
            data = self._load_hdf5(filepath)
        elif format == "npz":
            data = self._load_npz(filepath)
        elif format == "json":
            data = self._load_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Reconstruct dataset object
        if 'next_observations' in data and data['next_observations'] is not None:
            # EnhancedImitationDataset format
            from expert_data_collector import EnhancedImitationDataset
            dataset = EnhancedImitationDataset()

            # Restore data
            dataset.observations = data['observations']
            dataset.actions = data['actions']
            dataset.rewards = data['rewards']
            dataset.next_observations = data.get('next_observations', [])
            dataset.dones = data.get('dones', [])
            dataset.episode_ids = data.get('episode_ids', [])
            dataset.step_ids = data.get('step_ids', [])
            dataset.collection_methods = data.get('collection_methods', [])
            dataset.metadata = data.get('metadata', [])
            dataset.current_size = len(dataset.observations)

        else:
            # ExpertDataset format
            dataset = ExpertDataset(
                observations=data['observations'],
                actions=data['actions'],
                rewards=data.get('rewards', None),
                metadata=data.get('metadata', {})
            )

        print(f"[DataManager] Loaded dataset with {len(dataset)} transitions")
        return dataset

    def create_dataloader(self, dataset: Union['EnhancedImitationDataset', ExpertDataset],
                         batch_size: int = 32, shuffle: bool = True,
                         num_workers: int = 0, pin_memory: bool = True,
                         filter_method: Optional[str] = None) -> DataLoader:
        """
        Create PyTorch DataLoader from dataset.

        Args:
            dataset: Dataset to create loader from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            filter_method: Filter by collection method (for EnhancedImitationDataset)

        Returns:
            PyTorch DataLoader
        """
        # Convert to ExpertDataset if needed
        if not isinstance(dataset, ExpertDataset):
            # Convert EnhancedImitationDataset to ExpertDataset
            pytorch_dataset = ExpertDataset(
                observations=dataset.observations,
                actions=dataset.actions,
                rewards=getattr(dataset, 'rewards', None),
                metadata=getattr(dataset, 'metadata', {})
            )
        else:
            pytorch_dataset = dataset

        return DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def merge_datasets(self, dataset_paths: List[str], output_path: str,
                      format: str = "pickle") -> str:
        """
        Merge multiple datasets into one.

        Args:
            dataset_paths: List of paths to datasets to merge
            output_path: Path for merged dataset
            format: Output format

        Returns:
            Path to merged dataset
        """
        print(f"[DataManager] Merging {len(dataset_paths)} datasets...")

        # Load all datasets
        datasets = []
        for path in dataset_paths:
            try:
                dataset = self.load_dataset(path)
                datasets.append(dataset)
                print(f"[DataManager] Loaded {path}: {len(dataset)} transitions")
            except Exception as e:
                print(f"[DataManager] Warning: Failed to load {path}: {e}")

        if not datasets:
            raise ValueError("No datasets successfully loaded")

        # Create merged dataset
        from expert_data_collector import EnhancedImitationDataset
        merged = EnhancedImitationDataset()

        episode_offset = 0
        for i, dataset in enumerate(datasets):
            print(f"[DataManager] Merging dataset {i+1}/{len(datasets)}...")

            if hasattr(dataset, 'observations'):
                # EnhancedImitationDataset
                for j in range(len(dataset.observations)):
                    merged.add_transition(
                        obs=dataset.observations[j],
                        action=dataset.actions[j],
                        reward=dataset.rewards[j] if dataset.rewards else 0.0,
                        next_obs=dataset.next_observations[j] if dataset.next_observations else dataset.observations[j],
                        done=dataset.dones[j] if dataset.dones else False,
                        episode_id=(dataset.episode_ids[j] if dataset.episode_ids else 0) + episode_offset,
                        step_id=dataset.step_ids[j] if dataset.step_ids else j,
                        collection_method=dataset.collection_methods[j] if dataset.collection_methods else f"dataset_{i}",
                        metadata=dataset.metadata[j] if dataset.metadata else {}
                    )

                # Update episode offset
                if dataset.episode_ids:
                    episode_offset = max(dataset.episode_ids) + 1

            else:
                # ExpertDataset
                for j in range(len(dataset)):
                    obs, action, reward = dataset[j]
                    merged.add_transition(
                        obs=obs.numpy(),
                        action=action.numpy(),
                        reward=reward,
                        next_obs=obs.numpy(),  # Use same obs as next_obs
                        done=False,
                        episode_id=episode_offset + j // 1000,  # Estimate episodes
                        step_id=j % 1000,
                        collection_method=f"dataset_{i}",
                        metadata={}
                    )

        # Save merged dataset
        output_filepath = self.save_dataset(merged, output_path, format)
        print(f"[DataManager] Merged dataset saved: {len(merged)} total transitions")

        return output_filepath

    def split_dataset(self, dataset_path: str, train_ratio: float = 0.8,
                     output_prefix: str = "split", format: str = "pickle") -> Tuple[str, str]:
        """
        Split dataset into train/validation sets.

        Args:
            dataset_path: Path to dataset to split
            train_ratio: Ratio for training set
            output_prefix: Prefix for output files
            format: Output format

        Returns:
            Tuple of (train_path, val_path)
        """
        dataset = self.load_dataset(dataset_path)
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)

        print(f"[DataManager] Splitting dataset: {train_size} train, {total_size - train_size} val")

        # Create indices
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create split datasets
        from expert_data_collector import EnhancedImitationDataset

        train_dataset = EnhancedImitationDataset()
        val_dataset = EnhancedImitationDataset()

        # Populate train dataset
        for idx in train_indices:
            if hasattr(dataset, 'observations'):
                train_dataset.add_transition(
                    obs=dataset.observations[idx],
                    action=dataset.actions[idx],
                    reward=dataset.rewards[idx] if dataset.rewards else 0.0,
                    next_obs=dataset.next_observations[idx] if dataset.next_observations else dataset.observations[idx],
                    done=dataset.dones[idx] if dataset.dones else False,
                    episode_id=dataset.episode_ids[idx] if dataset.episode_ids else idx // 1000,
                    step_id=dataset.step_ids[idx] if dataset.step_ids else idx,
                    collection_method=dataset.collection_methods[idx] if dataset.collection_methods else "train",
                    metadata=dataset.metadata[idx] if dataset.metadata else {}
                )

        # Populate val dataset
        for idx in val_indices:
            if hasattr(dataset, 'observations'):
                val_dataset.add_transition(
                    obs=dataset.observations[idx],
                    action=dataset.actions[idx],
                    reward=dataset.rewards[idx] if dataset.rewards else 0.0,
                    next_obs=dataset.next_observations[idx] if dataset.next_observations else dataset.observations[idx],
                    done=dataset.dones[idx] if dataset.dones else False,
                    episode_id=dataset.episode_ids[idx] if dataset.episode_ids else idx // 1000,
                    step_id=dataset.step_ids[idx] if dataset.step_ids else idx,
                    collection_method=dataset.collection_methods[idx] if dataset.collection_methods else "val",
                    metadata=dataset.metadata[idx] if dataset.metadata else {}
                )

        # Save split datasets
        train_path = self.save_dataset(train_dataset, f"{output_prefix}_train", format)
        val_path = self.save_dataset(val_dataset, f"{output_prefix}_val", format)

        return train_path, val_path

    def get_dataset_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about a dataset without fully loading it.

        Args:
            filepath: Path to dataset

        Returns:
            Dataset information dictionary
        """
        filepath = Path(filepath)
        format = self._detect_format(filepath)

        info = {
            'filepath': str(filepath),
            'format': format,
            'file_size_mb': filepath.stat().st_size / (1024 * 1024),
            'last_modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
        }

        try:
            if format == "pickle":
                # Quick pickle inspection
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        info['dataset_size'] = len(data.get('observations', []))
                        info['save_metadata'] = data.get('save_metadata', {})

            elif format == "hdf5":
                # HDF5 inspection
                with h5py.File(filepath, 'r') as f:
                    if 'observations' in f:
                        info['dataset_size'] = len(f['observations'])
                    if 'save_metadata' in f.attrs:
                        info['save_metadata'] = json.loads(f.attrs['save_metadata'])

        except Exception as e:
            info['error'] = str(e)

        return info

    def list_datasets(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all datasets in a directory.

        Args:
            directory: Directory to search (uses base_dir if None)

        Returns:
            List of dataset information dictionaries
        """
        search_dir = Path(directory) if directory else self.base_dir

        dataset_files = []
        for ext in ['.pkl', '.h5', '.npz', '.json']:
            dataset_files.extend(search_dir.glob(f"*{ext}"))

        datasets = []
        for filepath in sorted(dataset_files):
            try:
                info = self.get_dataset_info(filepath)
                datasets.append(info)
            except Exception as e:
                print(f"[Warning] Could not get info for {filepath}: {e}")

        return datasets

    # Private methods for format-specific operations

    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()
        if suffix == '.pkl':
            return 'pickle'
        elif suffix == '.h5':
            return 'hdf5'
        elif suffix == '.npz':
            return 'npz'
        elif suffix == '.json':
            return 'json'
        else:
            raise ValueError(f"Cannot detect format for file: {filepath}")

    def _save_pickle(self, data: Dict[str, Any], filepath: Path, compress: bool):
        """Save data in pickle format."""
        if compress:
            with gzip.open(str(filepath) + '.gz', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, filepath: Path) -> Dict[str, Any]:
        """Load data from pickle format."""
        if filepath.suffix == '.gz' or str(filepath).endswith('.pkl.gz'):
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)

    def _save_hdf5(self, data: Dict[str, Any], filepath: Path, compress: bool):
        """Save data in HDF5 format."""
        compression = 'gzip' if compress else None

        with h5py.File(filepath, 'w') as f:
            # Save arrays
            for key, value in data.items():
                if key == 'save_metadata':
                    f.attrs['save_metadata'] = json.dumps(value)
                elif isinstance(value, (list, np.ndarray)) and value is not None:
                    if len(value) > 0:
                        if isinstance(value[0], np.ndarray):
                            # Stack arrays for efficient storage
                            stacked = np.stack(value)
                            f.create_dataset(key, data=stacked, compression=compression)
                        else:
                            # Regular list
                            f.create_dataset(key, data=value, compression=compression)

    def _load_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """Load data from HDF5 format."""
        data = {}
        with h5py.File(filepath, 'r') as f:
            # Load datasets
            for key in f.keys():
                dataset = f[key]
                if len(dataset.shape) > 1 and dataset.shape[0] > 0:
                    # Convert back to list of arrays
                    data[key] = [dataset[i] for i in range(dataset.shape[0])]
                else:
                    data[key] = dataset[...]

            # Load metadata
            if 'save_metadata' in f.attrs:
                data['save_metadata'] = json.loads(f.attrs['save_metadata'])

        return data

    def _save_npz(self, data: Dict[str, Any], filepath: Path, compress: bool):
        """Save data in NPZ format."""
        save_dict = {}

        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    save_dict[key] = np.stack(value)
                else:
                    save_dict[key] = np.array(value)
            elif isinstance(value, dict):
                save_dict[f"{key}_json"] = np.array([json.dumps(value)])

        if compress:
            np.savez_compressed(filepath, **save_dict)
        else:
            np.savez(filepath, **save_dict)

    def _load_npz(self, filepath: Path) -> Dict[str, Any]:
        """Load data from NPZ format."""
        npz_data = np.load(filepath, allow_pickle=True)
        data = {}

        for key, value in npz_data.items():
            if key.endswith('_json'):
                original_key = key[:-5]  # Remove '_json' suffix
                data[original_key] = json.loads(value.item())
            else:
                # Convert back to list if it was originally a list of arrays
                if len(value.shape) > 1:
                    data[key] = [value[i] for i in range(value.shape[0])]
                else:
                    data[key] = value.tolist()

        return data

    def _save_json(self, data: Dict[str, Any], filepath: Path, compress: bool):
        """Save data in JSON format (metadata only, not suitable for large arrays)."""
        # Only save metadata and statistics, not actual data arrays
        json_data = {}

        for key, value in data.items():
            if isinstance(value, (dict, str, int, float, bool)) or value is None:
                json_data[key] = value
            elif isinstance(value, list) and len(value) > 0:
                # Only save first few items as examples
                if isinstance(value[0], (int, float, bool, str)):
                    json_data[f"{key}_sample"] = value[:10]  # First 10 items
                    json_data[f"{key}_length"] = len(value)
                else:
                    json_data[f"{key}_length"] = len(value)
                    json_data[f"{key}_type"] = str(type(value[0]))

        if compress:
            with gzip.open(str(filepath) + '.gz', 'wt', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)

    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load data from JSON format."""
        if filepath.suffix == '.gz' or str(filepath).endswith('.json.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)


def main():
    """Demo and CLI interface for data utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="Data Management Utilities")
    parser.add_argument('--command', choices=['info', 'list', 'convert', 'merge', 'split'],
                       required=True, help='Command to execute')
    parser.add_argument('--input', help='Input file/directory path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['pickle', 'hdf5', 'npz', 'json'],
                       default='pickle', help='Output format')
    parser.add_argument('--compress', action='store_true', help='Compress output')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio for split')

    args = parser.parse_args()

    dm = DataManager()

    if args.command == 'info':
        if not args.input:
            print("--input required for info command")
            return
        info = dm.get_dataset_info(args.input)
        print(json.dumps(info, indent=2, default=str))

    elif args.command == 'list':
        datasets = dm.list_datasets(args.input)
        for dataset in datasets:
            print(f"{dataset['filepath']}: {dataset.get('dataset_size', 'unknown')} transitions")

    elif args.command == 'convert':
        if not args.input or not args.output:
            print("--input and --output required for convert command")
            return
        dataset = dm.load_dataset(args.input)
        dm.save_dataset(dataset, args.output, args.format, args.compress)

    elif args.command == 'merge':
        print("Merge command requires multiple input files - implement as needed")

    elif args.command == 'split':
        if not args.input or not args.output:
            print("--input and --output required for split command")
            return
        train_path, val_path = dm.split_dataset(args.input, args.train_ratio, args.output, args.format)
        print(f"Split complete: {train_path}, {val_path}")


if __name__ == "__main__":
    main()