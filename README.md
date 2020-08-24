This is a Face Mask detector using tensorflow lite model. I have used SSD Mobilenet-V2 Quantized COCO model from the tf1 model zoo for training the model and converting it into a .tflite format.

Run the detect_image.py script to see the code in action. The beauty of this project is that the model is only 4.5MB which can easily be used with the android phone or a raspberry pi.

The problem which i really face is when i do the detection through a webcam or a video, One single frame is processed at about ~2 secands. For a real time application that is High Latency and not accepted.
