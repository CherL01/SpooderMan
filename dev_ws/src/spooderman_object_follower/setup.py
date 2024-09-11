from setuptools import find_packages, setup

package_name = 'spooderman_object_follower'

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
    maintainer_email='cherry.lian@gatech.edu',
    description='Object follower turtlebot3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'view_image_raw = spooderman_object_follower.view_image_raw:main'
        ],
    },
)
