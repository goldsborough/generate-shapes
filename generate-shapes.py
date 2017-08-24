#!/usr/bin/env python3

import argparse
import json
import os
import time

import numpy as np
import scipy.misc

NUMBER_OF_ATTEMPS_TO_FIT_SHAPES = 1000


def make_label(shape_name, x_0, width, y_0, height):
    return {
        'class': shape_name,
        'x1': x_0,
        'x2': x_0 + width,
        'y1': y_0,
        'y2': y_0 + height,
    }


def generate_rectangle(x_0, y_0, width, height, color, min_dimension,
                       max_dimension):
    # (x_0, y_0) is the top left corner
    available_width = min(width - x_0, max_dimension)
    if available_width <= min_dimension:
        raise ArithmeticError
    available_height = min(height - y_0, max_dimension)
    if available_height <= min_dimension:
        raise ArithmeticError
    w = np.random.randint(min_dimension, available_width)
    h = np.random.randint(min_dimension, available_height)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[y_0:y_0 + h, x_0:x_0 + w] = color
    label = make_label('rectangle', x_0, width, y_0, height)

    return mask, label


def generate_circle(x_0, y_0, width, height, color, min_dimension,
                    max_dimension):
    # (x_0, y_0) is the center
    left = x_0
    right = width - x_0
    top = y_0
    bottom = height - y_0
    available_radius = min(left, right, top, bottom, max_dimension)
    if available_radius <= min_dimension:
        raise ArithmeticError
    radius = np.random.randint(min_dimension, available_radius)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(x_0 - radius, x_0 + radius + 1):
        y = int(np.sqrt(radius**2 - (x - x_0)**2))
        mask[y_0 - y:y_0 + y + 1, x] = color
    diameter = 2 * radius
    label = make_label('circle', x_0 - radius, diameter, y_0 - radius,
                       diameter)

    return mask, label


def generate_triangle(x_0, y_0, width, height, color, min_dimension,
                      max_dimension):
    # (x_0, y_0) is the bottom left corner
    # We're making an equilateral triangle. Pick the side as the min of the
    # available space to the right and available space above.
    available_side = min(width - x_0, y_0 + 1, max_dimension)
    if available_side < min_dimension:
        raise ArithmeticError
    side = np.random.randint(min_dimension, available_side)
    slope = np.sqrt(3)  # this pops up after some math
    y = y_0
    mid_point = x_0 + side / 2
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    # Drawing a linear function with positive slope to the right up to the
    # mid-point, then drawing a linear function with negative slope.
    for x in range(x_0, x_0 + side):
        mask[int(y):y_0, x] = color
        if x < mid_point:
            # Subtracting y means going up ((0, 0) is top left)
            y -= slope
        else:
            y += slope
    label = make_label('triangle', x_0, width, y_0 - height, height)

    return mask, label


SHAPE_GENERATORS = dict(
    rectangle=generate_rectangle,
    circle=generate_circle,
    triangle=generate_triangle)


def pick_color(gray, min_intensity):
    size = 1 if gray else 3
    return np.random.randint(min_intensity, 255, size=size)


def generate_image(width, height, number_of_shapes, min_dimension,
                   max_dimension, gray, shape, min_intensity, allow_overlap):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    labels = []
    for _ in range(number_of_shapes):
        x = np.random.randint(width)
        y = np.random.randint(width)
        color = pick_color(gray, min_intensity)
        if shape is None:
            choices = list(SHAPE_GENERATORS.values())
            shape_generator = np.random.choice(choices)
        else:
            shape_generator = SHAPE_GENERATORS[shape]
        try:
            mask, label = shape_generator(x, y, width, height, color,
                                          min_dimension, max_dimension)
        except ArithmeticError:
            pass
        else:
            assert mask.sum() > 0
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or image[mask.nonzero()].min() == 255:
                image -= mask
                labels.append(label)

    return image, dict(boxes=labels)


def generate_dataset(number, width, height, min_shapes, max_shapes,
                     min_dimension, max_dimension, gray, shape, min_intensity,
                     allow_overlap):
    images = []
    labels = []
    for _ in range(number):
        for _ in range(NUMBER_OF_ATTEMPS_TO_FIT_SHAPES):
            number_of_shapes = np.random.randint(min_shapes, max_shapes + 1)
            image, image_labels = generate_image(
                width, height, number_of_shapes, min_dimension, max_dimension,
                gray, shape, min_intensity, allow_overlap)
            # Try again if we didn't fit any shapes for this image.
            if image_labels['boxes']:
                images.append(image)
                labels.append(image_labels)
                break
        else:
            print('Failed to fit any shapes for image! '
                  'Consider reducing the minimum dimension.')
    return images, labels


def save_images_and_labels(output_directory, images, labels):
    if os.path.exists(output_directory):
        assert os.path.isdir(output_directory)
    else:
        os.makedirs(output_directory)
    print('Saving to {0} ...'.format(os.path.abspath(output_directory)))
    for number, image in enumerate(images):
        path = os.path.join(output_directory, '{0}.png'.format(number))
        scipy.misc.imsave(path, image)
    labels_file_path = os.path.join(output_directory, 'labels.json')
    with open(labels_file_path, 'w') as labels_file:
        json.dump(labels, labels_file, indent=4)


def show_images(images):
    for image in images:
        scipy.misc.toimage(image).show()


def parse():
    parser = argparse.ArgumentParser(
        description='Generate Toy Object Detection Dataset')
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        required=True,
        help='The number of images to generate')
    parser.add_argument(
        '--width',
        type=int,
        default=128,
        help='The width of generated images (128)')
    parser.add_argument(
        '--height',
        type=int,
        default=128,
        help='The height of generated images (128)')
    parser.add_argument(
        '--max-shapes',
        type=int,
        default=10,
        help='The maximum number of shapes per image (10)')
    parser.add_argument(
        '--min-shapes',
        type=int,
        default=1,
        help='The maximum number of shapes per image (1)')
    parser.add_argument(
        '--min-dimension',
        type=int,
        default=10,
        help='The minimum dimension of a shape (10)')
    parser.add_argument(
        '--max-dimension',
        type=int,
        help='The maximum dimension of a shape (None)')
    parser.add_argument(
        '--min-intensity',
        type=int,
        default=128,
        help='The minimum intensity (0-255) for a pixel channel (128)')
    parser.add_argument(
        '--gray', action='store_true', help='Make all shapes grayscale')
    parser.add_argument(
        '--shape',
        choices=SHAPE_GENERATORS.keys(),
        help='Generate only this kind of shape')
    parser.add_argument(
        '-o', '--output-dir', help='The output directory where to save images')
    parser.add_argument(
        '--allow-overlap',
        action='store_true',
        help='Allow shapes to overlap on images')

    options = parser.parse_args()
    if options.max_dimension is None:
        options.max_dimension = max(options.height, options.width)

    if options.max_dimension - options.min_dimension < 2:
        raise RuntimeError('Available dimension would be too small')
    if (options.min_dimension >= options.width
            or options.min_dimension >= options.height):
        raise RuntimeError(
            'Minimum dimension must be less than width and height')
    if not (0 <= options.min_intensity <= 255):
        raise RuntimeError('Minimum intensity must be in interval [0, 255]')

    return options


def main():
    options = parse()
    start = time.time()
    images, labels = generate_dataset(
        options.number, options.width, options.height, options.min_shapes,
        options.max_shapes, options.min_dimension, options.max_dimension,
        options.gray, options.shape, options.min_intensity,
        options.allow_overlap)
    end = time.time() - start
    print('Generated {0} images in {1:.2f}s'.format(len(images), end))
    if options.output_dir is None:
        show_images(images)
    else:
        save_images_and_labels(options.output_dir, images, labels)


if __name__ == '__main__':
    main()
