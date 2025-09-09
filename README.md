# Pianist

## Generate USD file

```bash
uv run ./scripts/convert_urdf_to_usd.py ./source/pianist/data/robots/shadow_hand/urdf/shadow_hand_left.urdf ./source/pianist/data/robots/shadow_hand/usd/left_hand.usd --fix-base
uv run ./scripts/convert_urdf_to_usd.py ./source/pianist/data/robots/shadow_hand/urdf/shadow_hand_left_translation.urdf ./source/pianist/data/robots/shadow_hand/usd/left_hand_translation.usd --fix-base
```

## Preview environment

```bash
uv run ./scripts/zero_agent.py --task Shadow-Direct-v0 --num_envs 4
```