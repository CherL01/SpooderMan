from setuptools import find_packages, setup
import os  
from glob import glob

package_name = 'spooderman_navigate_to_goal'

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
    description='Goal navigation turtlebot3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'get_global_position = spooderman_navigate_to_goal.get_global_position:main',
            'go_to_goal = spooderman_navigate_to_goal.go_to_goal:main',
        ],
    },
)
