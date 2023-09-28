#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import os
import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.messages= None
        self.transform_counter = 0
        self.transform_mat = None

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        # Calculated intrinsic params after calibration
        # self.intrinsic_matrix = np.array([[939.409, 0, 618.505],
        #                                   [0, 945.988, 328.224],
        #                                   [0, 0, 1]])
        # Factory Intrinsic Setting
        self.intrinsic_matrix = np.array([[907.9, 0, 641.046],
                                          [0, 908.35, 354.884],
                                          [0, 0, 1]])
        inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        
        return inv_intrinsic_matrix

    def homogeneous_transform_mat(self):
        """!
        @brief      Return the homogeneous transform matrix for the camera to world coordinates.

        """
        # This is Hinv
        # inv_matrix_homogeneous = np.array([[1, 0, 0, -40],
        #                                [0, -1, 0, 190],
        #                                [0, 0, -1, 1020],
        #                                [0, 0, 0, 1]])
        
        # matrix_homogeneous = np.array([[ 9.98526583e-01, -1.07363350e-02, -5.31920445e-02,  2.67844599e+01],
        #                                [-4.47806721e-04, -9.81829037e-01,  1.89767073e-01,  1.70920548e+02],
        #                                [-5.42628967e-02, -1.89463647e-01, -9.80387201e-01,  1.03091259e+03],
        #                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        matrix_homogeneous = np.load(file=os.path.join('extrinsic.npy'))
        if matrix_homogeneous is not None:
            matrix_homogeneous = np.load(file=os.path.join('extrinsic.npy'))
            matrix_homogeneous_inv = np.linalg.inv(matrix_homogeneous)
            return matrix_homogeneous, matrix_homogeneous_inv
        
    
    def extrinsic_matrix_cal(self):
        """!
        @brief      Solve for the extrinsic matrix of the camera.

        @param      intrinsic_matrix, distortion and image and camera points.
        """

        distortion = np.array([0.0791, -0.1517, -0.0029, 0.0027, 0]).astype(np.float32)
        # distortion = np.zeros(shape=(5,), dtype=np.float32)
        object_points_dict = {4: np.array([[-250, 275, 0], [-265, 260, 0], [-255, 260, 0], [-255, 290, 0], [-275, 290, 0]]),
                              3: np.array([[250, 275, 0], [235, 260, 0], [265, 260, 0], [265, 290, 0], [235, 290, 0]]), 
                              2: np.array([[250, -25, 0], [235, -40, 0], [265, -40, 0], [265, -10 ,0], [235, -10, 0]]),
                            #   6: np.array([[-425, 150, 155], [-440, 135, 155], [-410, 135, 155], [-410, 165, 155], [-440, 165, 155]]),
                              7: np.array([[-250, -25, 0], [-265, -40, 0], [-255, -40, 0], [-255, -10, 0], [-275, -10, 0]]),
                            #   8: np.array([[425, 150, 93], [410, 135, 93], [440, 135, 93], [440, 165, 93], [410, 165, 93]])
                             }

        # object_points = np.array([[250, -25, 0], [250, 275, 0], [-250, 275, 0], [-250, -25, 0]])
        count = 0
        image_points = []

        if self.messages is not None:
            for msg in self.messages.detections:
                count += 1
                id_list = [2, 3, 4, 6, 7, 8]
                if msg.id in id_list:
                    # The values from the tag_IDs are pixel coordinates of the camera.
                    u, v = int(msg.centre.x), int(msg.centre.y)
                    corn1_u, corn1_v = int(msg.corners[0].x), int(msg.corners[0].y)
                    corn2_u, corn2_v = int(msg.corners[1].x), int(msg.corners[1].y)
                    corn3_u, corn3_v = int(msg.corners[2].x), int(msg.corners[2].y)
                    corn4_u, corn4_v = int(msg.corners[3].x), int(msg.corners[3].y)
                    append_list = [u, v, corn1_u, corn1_v, corn2_u, corn2_v, corn3_u, corn3_v, corn4_u ,corn4_v]
                    # append_list = [u, v]
                    image_points.extend(append_list)
                    
                else:
                    raise NotImplementedError
                
                if count > 20:
                    break
            
            array_list = []
            for key in sorted(object_points_dict.keys()):
                array_list.append(object_points_dict[key])
            
            object_points = np.concatenate(array_list, axis=0)
            object_points = object_points.astype(np.float32)

            image_points = np.array(image_points).astype(np.float32)
            # print(f"image_points: {image_points.shape}")
            image_points = image_points.reshape(20, 2)

            # Gives us the H matrix
            _, rvec, tvec = cv2.solvePnP(object_points, image_points, self.intrinsic_matrix, distortion)
            rmatrix = cv2.Rodrigues(rvec)[0]
            H_transform = np.hstack((rmatrix, tvec.reshape(3,1)))
            H_transform = np.vstack((H_transform,np.array([0,0,0,1])))

            return H_transform
    
    def blockDetector(self, image):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """

        # image = self.VideoFrame.copy()
        # cv2.imread(image)
        cv2.imwrite("image.jpg", image)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        cv2.imwrite("rgb_img.jpg", rgb_image)
        # time.sleep(2)

        cv2.imwrite("hsv_img.jpg", hsv_image)
        # time.sleep(2)

        
        # range for blue 
        lower_blue = np.array([90, 150, 20])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        detected_blue_output = cv2.bitwise_and(image, image, mask = mask_blue)
        cv2.imwrite("blue_color_detection.jpg", detected_blue_output)
        # time.sleep(2)


        # Range for orange
        orange_lower = np.array([5,120,20])
        orange_upper = np.array([20,255,255])
        mask_orange = cv2.inRange(hsv_image, orange_lower, orange_upper)
        detected_orange_output = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_orange)
        cv2.imwrite("orange_color_detection.jpg", detected_orange_output)
        # time.sleep(2)

        # Range for upper range
        red1_lower = np.array([170,110,20])
        red1_upper = np.array([180,255,255])
        mask_red1 = cv2.inRange(hsv_image, red1_lower, red1_upper)


        # red_lower = np.array([0,110,20])
        # red_upper = np.array([5,255,255])
        # mask_red2 = cv2.inRange(hsv_image, red_lower, red_upper)

        # mask_red  = mask_red1 + mask_red2
        
        detected_red_output = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_red1)
        cv2.imwrite("red_color_detection.png", detected_red_output)

        yellow_lower = np.array([23, 75, 50])
        yellow_upper = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
        detected_yellow_output = cv2.bitwise_and(rgb_image, rgb_image, mask = mask_yellow)
        cv2.imwrite("yellow_color_detection.jpg", detected_yellow_output)
        
        green_lower = np.array([40,50,20])
        green_upper = np.array([86,255,255])
        mask_green = cv2.inRange(hsv_image, green_lower, green_upper)
        detected_green_output = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_green)
        cv2.imwrite("green_color_detection.jpg", detected_green_output)

        purple_lower = np.array([120,40,0])
        purple_upper = np.array([150,255,255])
        mask_purple = cv2.inRange(hsv_image, purple_lower, purple_upper)
        detected_purple_output = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_purple)
        cv2.imwrite("purple_color_detection.png", detected_purple_output)


        mask = mask_yellow + mask_blue + mask_green + mask_orange + mask_purple + mask_red1
        detected_colors = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
        cv2.imwrite("detection.png", detected_colors)
        
        return


    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        if self.messages is not None:
            if self.transform_counter < 5:
                self.transform_counter += 1
                self.GridFrame = self.VideoFrame.copy()
                x_points, y_points = self.grid_points # these points are  in objects points in the world frame 
                depth_value = 0
                world_coordinates = [[x,y, depth_value, 1] for x,y in zip(x_points.ravel(), y_points.ravel())]
                world_coordinates_array = np.array(world_coordinates)

                # world to camera coordinates
                mat_h, mat_hinv = self.homogeneous_transform_mat()
                camera_coordinates = mat_h @ np.transpose(world_coordinates_array)
                # print(f"camera_coord {camera_coordinates[0]}")


                # camera to pixel coordinates
                pixel_coord = self.intrinsic_matrix @ camera_coordinates[:3, :]
                pixel_coord /= pixel_coord[2][:]
                # print(f"Pixel coordinates shape {pixel_coord.shape} and first item {pixel_coord[0][0]}")

                # plotting on the video frame 
                x_coords = pixel_coord[0, :].astype(int)
                y_coords = pixel_coord[1, :].astype(int)  

                for x, y in zip(x_coords, y_coords):
                    cv2.circle(self.GridFrame, (x, y), radius= 2, color=(0, 255, 0), thickness=2)           

                projected_points = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32)
                original_points = np.array([[174, 53], [1164, 53], [1111, 658], [227, 659]], dtype=np.float32)
                self.transform_mat = cv2.getPerspectiveTransform(original_points, projected_points)

                self.GridFrame = cv2.warpPerspective(self.GridFrame, self.transform_mat, (1280, 720))
                self.blockDetector(self.GridFrame)
        return 
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        self.messages = msg
        for msg in msg.detections:

            # id_list = [2, 3, 4, 6, 7, 8]
            id_list = [2, 3, 4, 7]
            if msg.id in id_list:
                cv2.circle(modified_image, (int(msg.centre.x), int(msg.centre.y)), 1, (255, 0,0), 2)
                cv2.rectangle(modified_image, (int(msg.corners[0].x), int(msg.corners[0].y)), (int(msg.corners[2].x), int(msg.corners[2].y)), (0, 255, 0), 2)
                
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (int(msg.corners[3].x + 5), int(msg.corners[3].y - 25))
                fontScale              = 0.5
                fontColor              = (0,0,255)
                thickness              = 2
                lineType               = 2

                cv2.putText(modified_image, f"ID:{msg.id}", 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

            else:
                # raise NotImplementedError
                continue

        # print("Here")
        self.TagImageFrame = modified_image

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            print(f"cvdepth {cv_depth}")
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                # self.camera.blockDetector()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()