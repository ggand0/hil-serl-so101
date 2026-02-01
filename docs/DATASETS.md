# Dataset Setup

All dataset labels and merge configurations are stored in `data/labels/` as JSON files.

## Directory Structure

```
data/labels/
├── grasp_only_positive.json    # Source: 15 positive demo episodes
├── grasp_only_negative.json    # Source: 10 negative/mixed episodes
├── nighttime.json              # Source: 10 nighttime episodes
├── daytime_negative.json       # Source: 5 daytime negative episodes
├── reward_v1.json              # Merge config: positive + negative
├── reward_v3.json              # Merge config: positive + negative + daytime_negative
└── offline_v1.json             # Merge config: offline RL dataset
```

## Source Label Files

Each source file defines a dataset path and per-episode success frame labels.

### Label Format

```json
{
    "dataset_path": "/path/to/dataset",
    "episodes": {
        "0": {"success_start": 34},           // frames 34+ are success
        "1": {"ranges": [[40, 48], [60, null]]}, // frames 40-48 and 60+ are success
        "2": null,                             // all frames are failure
        "3": {"success_start": 27, "exclude": [0]} // frames 27+ success, exclude frame 0
    }
}
```

- `success_start`: All frames from this index onwards are success
- `ranges`: List of [start, end] pairs. `null` end means "to end of episode"
- `null`: All frames are failure (negative example)
- `exclude`: List of frame indices to mark as failure regardless of success_range

### Source Files

| File | Dataset Path | Episodes | Description |
|------|--------------|----------|-------------|
| `grasp_only_positive.json` | `~/.cache/.../so101_grasp_only_v1` | 0-14 | Successful grasp demos |
| `grasp_only_negative.json` | `~/.cache/.../so101_grasp_only_negative_v1` | 0-9 | Failed/mixed grasp attempts |
| `nighttime.json` | `data/grasp_only_reward_nighttime` | 0-9 | Nighttime lighting data |
| `daytime_negative.json` | `data/grasp_only_reward_daytime_negative` | 0-4 | Daytime negative examples |

## Merge Config Files

Merge configs reference source label files and define output dataset.

### Format

```json
{
    "output_repo_id": "gtgando/so101_grasp_only_reward_v3",
    "sources": ["grasp_only_positive", "grasp_only_negative", "daytime_negative"]
}
```

### Reward Classifier Datasets

| Config | Sources | Output | Use Case |
|--------|---------|--------|----------|
| `reward_v1.json` | positive + negative | `so101_grasp_only_reward_v1` | Original reward classifier |
| `reward_v3.json` | positive + negative + daytime_negative | `so101_grasp_only_reward_v3` | With daytime negatives |

### Offline RL Dataset

```json
{
    "output_repo_id": "gtgando/so101_grasp_only_offline_v1",
    "sources": [
        {
            "dataset_path": "/path/to/so101_grasp_only_v1",
            "episodes": [0, 1, 2, ..., 14]
        },
        {
            "dataset_path": "/path/to/so101_grasp_only_negative_v1",
            "episodes": [2, 7, 8]
        }
    ]
}
```

## Scripts

### Create Reward Classifier Dataset

```bash
python scripts/create_grasponly_reward_dataset.py --config reward_v1
python scripts/create_grasponly_reward_dataset.py --config reward_v3
```

### Create Offline RL Dataset

```bash
python scripts/create_grasponly_offline_dataset.py --config offline_v1
```

## Output Locations

All merged datasets are saved to `~/.cache/huggingface/lerobot/gtgando/`:

- `so101_grasp_only_reward_v1` - 25 episodes, 1927 frames
- `so101_grasp_only_reward_v3` - 30 episodes, 2926 frames
- `so101_grasp_only_offline_v1` - 18 episodes

## Adding New Data

1. Record new episodes using recording config
2. Extract frames: `ffmpeg -i video.mp4 frames/frame_%04d.jpg`
3. Label frames by identifying success start frame
4. Create/update source label JSON in `data/labels/`
5. Update or create merge config JSON
6. Run merge script with `--config` flag
