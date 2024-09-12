from setuptools import find_packages, setup

import os  
from glob import glob

package_name = 'spooderman_object_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cherry Lian',
    maintainer_email='cherry.lian@gatech.edu',
    description='Object follower turtlebot3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'find_object = spooderman_object_follower.find_object:main',
            'rotate_robot = spooderman_object_follower.rotate_robot:main',
        ],
    },
)
