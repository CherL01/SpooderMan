from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spooderman_chase_object',
            executable='detect_object',
            name='detect_object'
        ),
        Node(
            package='spooderman_chase_object',
            executable='get_object_range',
            name='get_object_range'
        ),
        Node(
            package='spooderman_chase_object',
            executable='chase_object',
            name='chase_object'
        )
    ])