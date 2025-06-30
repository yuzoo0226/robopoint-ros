# RoboPoint for ROS Noetic

## Environments

```bash
cd src/
pip install .
```

## launch vqa service

```bash
rosrun robopoint_ros robopoint_vqa_service.py
```

## service client

```bash
rosrun robopoint_ros lib_robopoint_vqa.py
```

## python usage

```python
import sys
import roslib
sys.path.append(roslib.packages.get_pkg_dir("robopoint_ros") + "/script")
from lib_tracking import CutieTrackingUtils

robopoint = RoboPointVQAUtils()
robopoint.call_get_place_pose(image_path="img_path.jpg", prompt="output place point")

```
