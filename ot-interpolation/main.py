import os
from src.io import read_mesh
import numpy as np
from test import (test_two_shapes_with_voxelization_tosca,
                  test_two_shapes_without_voxelization_tosca,
                  test_voxelization_tosca,
                  test_barycenter_two_images,
                  test_barycenter_two_shapes_tosca,
                  test_barycenter_three_shapes_tosca)


def main():
    # test_two_shapes_without_voxelization_tosca()
    # test_two_shapes_with_voxelization_tosca()
    # test_voxelization()
    # test_barycenter_two_images()
    test_barycenter_two_shapes_tosca()
    # test_barycenter_three_shapes_tosca()


if __name__=="__main__":
    main()
