#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
import numpy as np
import math
from tsr.generic import cylinder_grasp, box_grasp
from tf_transformations import quaternion_from_matrix


def random_pose(min_bounds, max_bounds):
    """
    Generate a random 4x4 homogeneous transform with translation
    uniform in [min_bounds, max_bounds] and identity rotation.
    """
    T = np.eye(4)
    T[0:3, 3] = np.random.uniform(min_bounds, max_bounds)
    return T


class TSRVisualizer(Node):
    def __init__(self):
        super().__init__('tsr_visualizer')
        # Publisher for MarkerArray
        self.publisher = self.create_publisher(MarkerArray, 'tsr_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_markers)

        # Declare parameters
        self.declare_parameter('object_type', 'cylinder')
        self.declare_parameter('cylinder_radius', 0.1)
        self.declare_parameter('cylinder_height', 0.4)
        self.declare_parameter('box_length', 0.4)
        self.declare_parameter('box_width', 0.2)
        self.declare_parameter('box_height', 0.6)
        self.declare_parameter('lateral_offset', 0.02)
        self.declare_parameter('vertical_tolerance', 0.05)
        self.declare_parameter('lateral_tolerance', 0.01)
        self.declare_parameter('mesh_resource', 'package://geodude_description/meshes/gripper.stl')

    def publish_markers(self):
        markers = MarkerArray()
        marker_id = 0

        # Read parameters
        object_type = self.get_parameter('object_type').get_parameter_value().string_value
        lateral_offset = self.get_parameter('lateral_offset').value
        mesh = self.get_parameter('mesh_resource').get_parameter_value().string_value

        # Generate a random object pose
        if object_type == 'cylinder':
            r = self.get_parameter('cylinder_radius').value
            h = self.get_parameter('cylinder_height').value
            vt = self.get_parameter('vertical_tolerance').value
            yaw_range = [-math.pi/2, math.pi/2]
            obj_pos = np.random.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, h])
            chains = cylinder_grasp(None, obj_pos, r, h,
                                    lateral_offset=lateral_offset,
                                    vertical_tolerance=vt,
                                    yaw_range=yaw_range,
                                    manip_idx=0)
        else:
            L = self.get_parameter('box_length').value
            W = self.get_parameter('box_width').value
            H = self.get_parameter('box_height').value
            tol = self.get_parameter('lateral_tolerance').value
            T = random_pose([-0.5, -0.5, 0.0], [0.5, 0.5, H])
            # Dummy box wrapper
            class DummyBox:
                def __init__(self, T): self._T = T
                def GetTransform(self): return self._T
            chains = box_grasp(None, DummyBox(T), L, W, H,
                               manip_idx=0,
                               lateral_offset=lateral_offset,
                               lateral_tolerance=tol)

        # For each TSRChain, sample a pose and create a mesh marker
        for chain in chains:
            # sample one valid end-effector transform from the chain
            Tworld = chain.sample()
            # Convert to quaternion
            q = quaternion_from_matrix(Tworld)

            # Build ROS2 Marker
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'tsr'
            marker.id = marker_id
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.mesh_resource = mesh
            marker.pose.position = Point(x=Tworld[0,3], y=Tworld[1,3], z=Tworld[2,3])
            marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            # Semi-transparent green
            marker.color.a = 0.8
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime.sec = 1

            markers.markers.append(marker)
            marker_id += 1

        # Publish all markers
        self.publisher.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = TSRVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
