from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mppi_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'),
            glob('mppi_controller/models/*.urdf')),
        (os.path.join('share', package_name, 'models', 'meshes'),
            glob('mppi_controller/models/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='ch3044637@khu.ac.kr',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_node = mppi_controller.controller_node:main'
        ],
    },
)
