# install mavsdk, pygame, and gstreamer

import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
import pygame
import cv2
import cv2.aruco as aruco
import gi
import numpy as np
import math

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# GStreamer-based video class for capturing drone video
class Video():
    def __init__(self, port=5600):
        Gst.init(None)
        self.port = port
        self._frame = None

        self.video_source = 'udpsrc port={}'.format(self.port)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        return self._frame

    def frame_available(self):
        return self._frame is not None

    def run(self):
        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame
        return Gst.FlowReturn.OK


# Initialize desired stable altitude at a specific level
target_altitude = 2.0

def maintain_altitude(drone, current_altitude, new_target_altitude):
    if current_altitude < new_target_altitude - 0.3:  # Drone is below target
        return -0.25  # Move up slowly
    elif current_altitude > new_target_altitude + 0.3:  # Drone is above target
        return 0.25  # Move down slowly
    return 0  # Maintain altitude


def process_frame(frame):
    # Convert the frame to HSV (Hue, Saturation, Value)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color ranges
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([175, 150, 150])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # Combine masks
    red_mask = mask1 + mask2

    # Find contours and return bounding box for the largest contour
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 200:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            return center_x, center_y, w, h  # Return bounding box coordinates and size
    return None  # No red box detected

# Main drone control
async def main():
    print("Connecting to drone...")
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()

    # Wait for the drone to reach a stable altitude
    await asyncio.sleep(10)

    # Initial setpoint before starting offboard mode
    initial_velocity = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    await drone.offboard.set_velocity_body(initial_velocity)

    print("-- Setting offboard mode")
    await drone.offboard.start()

    # Initialize GStreamer video object for capturing the drone's camera feed
    video = Video()

    detected_ids = []  # List to keep track of detected ArUco marker IDs

    # Main Control Loop
    while True:
        # Retrieve the current altitude
        async for position in drone.telemetry.position():
            current_altitude = position.relative_altitude_m
            break

        # If frame is available, process it
        if video.frame_available():
            frame = video.frame()
            frame = np.array(frame)
            box = process_frame(frame)

            if box:
                # Unpack bounding box coordinates
                x, y, w, h = box
                screen_center_x, screen_center_y = frame.shape[1] // 2, frame.shape[0] // 2

                altitude_alignment_threshold = frame.shape[0] // 20  # Pixel threshold for acceptable altitude alignment
                forward_threshold_low = 0.03 * frame.shape[0] * frame.shape[1]  # Lower boundary of lateral alignment
                forward_threshold_high = 0.05 * frame.shape[0] * frame.shape[1]  # Upper boundary of lateral alignment
                yaw_alignment_threshold = frame.shape[1] // 20  # Pixel threshold for acceptable yaw alignment

                # Adjust yaw
                if abs(x - screen_center_x) > yaw_alignment_threshold:
                    yaw_speed = -5 if x < screen_center_x else 5
                else:
                    yaw_speed = 0

                # Adjust altitude
                if abs(y - screen_center_y) > altitude_alignment_threshold:
                    down = -0.3 if y < screen_center_y else 0.3  # Adjust up or down
                    forward = 0  # Prevent forward movement until altitude matches
                    right = 0  # Prevent lateral movement until altitude matches

                else:
                    # Box is vertically aligned; proceed with forward/backward adjustments
                    down = 0  # Maintain current altitude
                    box_area = w * h
                    if box_area < forward_threshold_low:  # Too far, move forward
                        forward = 0.4
                    elif box_area > forward_threshold_high:  # Too close, move backward
                        forward = -0.4
                    else:
                        forward = 0  # Acceptable, maintain position

            else:
                # Rotate if no red box is detected
                yaw_speed = 30
                forward = 0
                right = 0
                down = maintain_altitude(drone, current_altitude, target_altitude)

            # Set drone velocity based on control decisions
            velocity = VelocityBodyYawspeed(forward, right, down, yaw_speed)
            await drone.offboard.set_velocity_body(velocity)

            # Display the processed frame
            cv2.imshow("Drone Camera Stream", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.1)

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())