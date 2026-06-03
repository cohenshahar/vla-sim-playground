from setuptools import setup
package_name = 'kr6_sim'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shahar',
    maintainer_email='shahar@example.com',
    description='KR6 MuJoCo simulation exposed as ROS2 nodes',
    license='MIT',
    entry_points={'console_scripts': [
        'sim_node = kr6_sim.sim_node:main',
    ]},
)
