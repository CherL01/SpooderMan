from setuptools import find_packages, setup
import os  
from glob import glob

package_name = 'spooderman_nav2goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cherry Lian',
    maintainer_email='cherrylian01@gmail.com',
    description='A package to navigate to a point in space',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigate_to_point = spooderman_nav2goal.navigate_to_point:main',
        ],
    },
)
