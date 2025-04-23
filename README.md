Virtual T-Shirt Try-On (Hand Gesture Based)
This project is a real-time virtual try-on application using a webcam. It allows users to switch between virtual T-shirts by pointing their index finger at on-screen arrows — just like interacting with touch buttons using hand gestures!
Features:
- AI-based body and hand detection using MediaPipe.
- Dynamic T-shirt overlay that fits automatically on your shoulders and torso.
- Touchless gesture interaction using your index finger to switch shirts.
- Mirror camera display for natural interaction.
- Real-time performance using only your laptop webcam (no special hardware needed).

Technologies Used
- Python
- OpenCV
- MediaPipe (Pose + Hands detection)
```bash
pip install opencv-python mediapipe numpy
python virtual_tryon.py
```
Make sure all images are placed in the same folder as the script.
