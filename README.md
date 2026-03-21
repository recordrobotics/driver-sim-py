# Realtime AdvantageScope Viewer

A lightweight Python viewer that renders the AdvantageScope 3D field and streams live data from NetworkTables.

## Requirements

- Python 3.12
- A GPU with OpenGL 3.3+ support

## Quick start

Install dependencies, then run the viewer:

```pwsh
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Configuration can be changed in [`config.json`](config.json)

The viewer connects to NetworkTables at `127.0.0.1` and subscribes to:

- `/AdvantageKit/RealOutputs/RobotModel/Robot`
- `/AdvantageKit/RealOutputs/RobotModel/MechanismPoses`
- `/AdvantageKit/RealOutputs/RobotModel/FuelPositions`

as well as other AdvantageKit values

## Notes

- Field and robot assets are loaded from `./assets`.
- AprilTag textures are pulled from `./textures`.
- If you add new assets, update `config.py` paths as needed.

## Troubleshooting

- If you see a blank window, confirm your GPU driver supports OpenGL 3.3+.
- If NetworkTables data is missing, check that AdvantageKit is publishing to localhost.
