from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spooderman_navigate_to_goal',
            executable='get_global_position',
            name='get_global_position'
        ),
        Node(
            package='spooderman_navigate_to_goal',
            executable='obstacle_detection',
            name='obstacle_detection'
        ),
        Node(
            package='spooderman_navigate_to_goal',
            executable='go_to_goal',
            name='go_to_goal'
        )
    ])