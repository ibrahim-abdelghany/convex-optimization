import numpy as np
import cvxpy as cp
from PIL import Image, ImageOps

### EE364a Homework 6 additional problems
# Exercise 5

def main():
    image = Image.open("hw6/flowgray.png")

    grey_image = ImageOps.grayscale(image)

    image_data = np.asarray(grey_image)

    K = np.heaviside(np.random.random(image_data.shape) - 0.5, 0).astype(np.uint8)

    output_image = interpolate_image_l2(image_data, K, norm=1)

    Image.fromarray(np.round(output_image).astype(np.uint8), mode='L').show()

    output_image = interpolate_image_l2(image_data, K, norm=2)

    Image.fromarray(np.round(output_image).astype(np.uint8), mode='L').show()

def interpolate_image_l2(image_data, K, norm=2):
    output_image = cp.Variable(image_data.shape)

    delta1 = output_image[:-1,:] - output_image[1:,:]
    delta2 = output_image[:,:-1] - output_image[:,1:]

    if norm == 1:
        objective = cp.sum_squares(delta1) + cp.sum_squares(delta2)
    elif norm == 2:
        objective = cp.sum(cp.abs(delta1)) + cp.sum(cp.abs(delta2))
    else:
        objective = 0

    constraints = [
        cp.multiply(K, output_image) == cp.multiply(K, image_data),
        output_image >= 0,
        output_image <= 255
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    result = problem.solve()

    return output_image.value

if __name__ == '__main__':
    main()