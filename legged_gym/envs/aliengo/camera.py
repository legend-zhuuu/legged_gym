import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from PIL import Image as im
import torch
from isaacgym import gymtorch


class Camera:
    def __init__(self, sim, envs):
        self.sim = sim
        self.envs = envs
        self.gym = gymapi.acquire_gym()
        self.camera_handles = [[]]
        self.camera_num = 1  # if robot use more camera, pls add camera handle
        self.init_camera_state()
        self.frame_count = 0

    def init_camera_state(self):
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 360
        camera_properties.height = 240
        camera_properties.enable_tensors = True

        for i in range(len(self.envs)):
            self.camera_handles.append([])
            h = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            camera_offset = gymapi.Vec3(1, 0,  0.)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(30))
            actor_handle = self.gym.get_actor_handle(self.envs[i], 0)
            body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], actor_handle, 0)

            self.gym.attach_camera_to_body(h, self.envs[i], body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(h)

            # # Set a fixed position and look-target for the first camera
            # # position and target location are in the coordinate frame of the environment
            # h1 = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            # camera_position = gymapi.Vec3(2, 10, 1.0)
            # camera_target = gymapi.Vec3(10, 10, 1.)
            # self.gym.set_camera_location(h1, self.envs[i], camera_position, camera_target)
            # self.camera_handles[i].append(h1)

    def get_picture(self):
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        camera_buffer = [[]]
        for i in range(len(self.envs)):
            camera_buffer.append([])
            for j in range(self.camera_num):
                # The gym utility to write images to disk is recommended only for RGB images.
                rgb_filename = "graphics_images/rgb_env%d_cam%d_frame%d.png" % (i, j, self.frame_count)

                # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
                # Here we retrieve a depth image, normalize it to be visible in an
                # output image and then write it to disk using Pillow
                # depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i][j], gymapi.IMAGE_DEPTH)
                depth_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i][j], gymapi.IMAGE_DEPTH)
                depth_image = gymtorch.wrap_tensor(depth_image)
                # -inf implies no depth value, set it to zero. output will be black.
                # depth_image[depth_image == -np.inf] = 0
                depth_image = torch.where(depth_image == -torch.inf, 0, depth_image)
                # clamp depth image to 10 meters to make output image human friendly
                depth_image = torch.where(depth_image < -10, -10, depth_image)

                # flip the direction so near-objects are light and far objects are dark
                normalized_depth = -255.0 * (depth_image / torch.min(depth_image + 1e-4))

                camera_buffer[i].append(normalized_depth)
        self.frame_count += 1
        return camera_buffer

    def save_img(self):
        if not os.path.exists("graphics_images"):
            os.mkdir("graphics_images")

        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        for i in range(min(4, len(self.envs))):
            for j in range(self.camera_num):
                # # The gym utility to write images to disk is recommended only for RGB images.
                # rgb_filename = "graphics_images/rgb_env%d_cam%d_frame%d.png" % (i, j, self.frame_count)
                # self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.camera_handles[i][j], gymapi.IMAGE_COLOR, rgb_filename)

                # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
                # Here we retrieve a depth image, normalize it to be visible in an
                # output image and then write it to disk using Pillow
                depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i][j], gymapi.IMAGE_DEPTH)

                # -inf implies no depth value, set it to zero. output will be black.
                depth_image[depth_image == -np.inf] = 0

                # clamp depth image to 10 meters to make output image human friendly
                depth_image[depth_image < -10] = -10

                # flip the direction so near-objects are light and far objects are dark
                normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))

                # Convert to a pillow image and write it to disk
                normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
                normalized_depth_image.save("graphics_images/depth_env%d_cam%d_frame%d.jpg" % (i, j, self.frame_count))
        self.frame_count += 1
