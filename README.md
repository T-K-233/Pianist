# Pianist

## Generate USD file

```bash
uv run ./scripts/tools/convert_urdf_to_usd.py ./source/pianist/data/robots/shadow_hand/urdf/shadow_hand_left.urdf ./source/pianist/data/robots/shadow_hand/usd/left_hand.usd --fix-base
uv run ./scripts/tools/convert_urdf_to_usd.py ./source/pianist/data/robots/shadow_hand/urdf/shadow_hand_left_translation.urdf ./source/pianist/data/robots/shadow_hand/usd/left_hand_translation.usd --fix-base
```

Generate piano
```bash
uv run ./scripts/tools/convert_piano_mjcf_to_urdf.py ./source/pianist/data/assets/piano/urdf/piano.xml
uv run ./scripts/tools/convert_urdf_to_usd.py ./source/pianist/data/assets/piano/urdf/piano.urdf ./source/pianist/data/assets/piano/usd/piano.usd --fix-base
```


## Preview environment

```bash
uv run ./scripts/zero_agent.py --task Piano-Shadow-v0 --num_envs 4
```

## Train

```bash
```