import numpy as np
from numpy.testing import assert_array_equal, assert_equal

import pytest
from generate_shapes import generate_shapes


def test_generates_color_images_with_correct_shape():
    images, _ = generate_shapes(
        number_of_images=100, width=128, height=128, max_shapes=1)

    assert len(images) == 100
    assert all(i.shape == (128, 128, 3) for i in images), images[0].shape


def test_generates_gray_images_with_correct_shape():
    images, _ = generate_shapes(
        number_of_images=100, width=128, height=128, max_shapes=1, gray=True)
    assert len(images) == 100
    assert all(i.shape == (128, 128, 1) for i in images), images[0].shape


def test_generates_correct_bounding_boxes_for_rectangles():
    (image, ), (labels, ) = generate_shapes(
        number_of_images=1,
        width=128,
        height=128,
        max_shapes=1,
        shape='rectangle')
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]

    # The crop is filled.
    assert (crop < 255).all()

    # The crop is complete.
    image[label.y1:label.y2, label.x1:label.x2] = 255
    assert (image == 255).all()


def test_generates_correct_bounding_boxes_for_triangles():
    (image, ), (labels, ) = generate_shapes(
        number_of_images=1,
        width=128,
        height=128,
        max_shapes=1,
        shape='triangle')
    # assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]

    # The crop is filled.
    assert (crop < 255).any()

    # The crop is complete.
    image[label.y1:label.y2, label.x1:label.x2] = 255
    assert (image == 255).all()


def test_generates_correct_bounding_boxes_for_circles():
    (image, ), (labels, ) = generate_shapes(
        number_of_images=1,
        width=43,
        height=44,
        max_shapes=1,
        min_dimension=20,
        max_dimension=20,
        shape='circle')
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]

    # The crop is filled.
    assert (crop < 255).any()

    # The crop is complete.
    image[label.y1:label.y2, label.x1:label.x2] = 255
    assert (image == 255).all()


def test_generate_circle_throws_when_dimension_too_small():
    with pytest.raises(ValueError):
        generate_shapes(
            number_of_images=1,
            width=64,
            height=128,
            max_shapes=1,
            min_dimension=1,
            max_dimension=1,
            shape='circle')


def test_generate_triangle_throws_when_dimension_too_small():
    with pytest.raises(ValueError):
        generate_shapes(
            number_of_images=1,
            width=128,
            height=64,
            max_shapes=1,
            min_dimension=1,
            max_dimension=1,
            shape='triangle')


def test_can_generate_one_by_one_rectangle():
    (image, ), (labels, ) = generate_shapes(
        number_of_images=1,
        width=50,
        height=128,
        max_shapes=1,
        min_dimension=1,
        max_dimension=1,
        shape='rectangle')
    assert len(labels) == 1
    label = labels[0]
    crop = image[label.y1:label.y2, label.x1:label.x2]
    assert (crop < 255).sum() == 3  # rgb
