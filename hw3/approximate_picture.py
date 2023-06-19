import cvxpy as cp
import numpy as np

from sign import sign

layer_transparency = 0.5

# def main():
#     approximate_picture(None,1)

oval_center = cp.Variable(2, pos=True)
oval_size = cp.Variable(2, pos=True)
oval_color = cp.Variable(1, pos=True)

errors = 0
# < 0 if pixel in oval, > 0 if pixel outside oval
pixel_location = np.array([1,1])

coefficient = cp.abs(oval_center - pixel_location)/oval_size
pixel_in_oval = cp.quad_form(coefficient, np.identity(2)) - 1
pixel_in_oval = cp.square((oval_center - pixel_location)/oval_size) - 1

# (oval_center0 - pixel_location0)^2 * oval_size1 + (oval_center1 - pixel_location1)^2 * oval_size0 - oval_size0*oval_size1 < 0

print(pixel_in_oval[0])
print(pixel_in_oval[0].curvature)
pixel_opacity = layer_transparency * (0.5 - 0.5 * sign(pixel_in_oval))
print(pixel_opacity.curvature)
layer_color = 0 + cp.multiply(cp.pos(oval_color[0]), cp.pos(pixel_opacity))
print(layer_color.curvature)
errors += cp.power(1 - layer_color, 2)


objective = cp.Minimize(cp.sign(pixel_in_oval))

constraints = [oval_color >= 0, oval_color <= 256]

problem = cp.Problem(objective, constraints)

problem.solve(qcp=True)

print(problem.value)

def approximate_picture(source_image, layers):
    target_picture = np.ones((10,10,3), dtype=np.int8)

    base_picture = np.zeros_like(target_picture)

    for i in range(layers):
        oval_center = cp.Variable(2)
        oval_size = cp.Variable(2, pos=True)
        oval_color = cp.Variable(3)
        
        errors = 0

        for row in range(np.size(target_picture, 0)):
            for column in range(np.size(target_picture, 1)):
                for color in range(np.size(target_picture, 2)):
                    # < 0 if pixel in oval, > 0 if pixel outside oval
                    pixel_in_oval = cp.square(cp.pos((oval_center[0] - row)/oval_size[0])) + cp.square(cp.pos(((oval_center[1] - column)/oval_size[1]))) - 1
                    pixel_opacity = layer_transparency * (0.5 - 0.5 * cp.sign(pixel_in_oval))
                    layer_color = base_picture[row, column] + oval_color[color] * pixel_opacity
                    errors += cp.power(color - layer_color, 2)
        
        objective = cp.Minimize(errors)

        constraints = [oval_color >= 0, oval_color <= 256]

        problem = cp.Problem(objective, constraints)

        problem.solve(qcp=True)

        print(problem.value)

if __name__ == "__main__":
    main()
