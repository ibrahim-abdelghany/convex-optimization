import cvxpy as cp

# (a) norm( [ x + 2*y , x - y ] ) == 0
# problem: norm is a convex function, cannot be used in equality constraint
# equivalent constraints: x + 2*y == 0, x - y == 0 (norm is zero when all items are zero)

x = cp.Variable();
y = cp.Variable();
z = cp.Variable();

problem = cp.Problem(cp.Minimize(x+y), [x + 2*y == 0, x - y == 0])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)

# (b) square( square( x + y ) ) <= x - y
# f * g (x) is convex iff f is convex, g is convex & f nondecreasing, here square(x+y) is decreasing for x+y <0 => violates composition rules
# equivalent constraints: (x+y)^4 <= x - y ; x+y is affine and nondecreasing, (.)^4 is convex

problem = cp.Problem(cp.Minimize(x), [(x + y)**4 <= x - y])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)


# (c) 1/x + 1/y <= 1; x >= 0; y >= 0
# this is a weighted sum of convex functions given positive domain 
# we use inv_pos(x) to turn the constraint into a convex function

problem = cp.Problem(cp.Minimize(x), [cp.inv_pos(x) + cp.inv_pos(y) <= 1])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)


# (d) norm([ max( x , 1 ) , max( y , 2 ) ]) <= 3*x + y
# norm of piecewise max violates monotonicity requirement
# equivalent to norm([x,y]) since norm <= norm[max] <= 3x+y

# wrong solution because 3*x + y needs to be bounded also by norm(1,2)

problem = cp.Problem(cp.Minimize(x), [cp.norm(cp.vstack([x, y])) <= 3*x+y])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)


# (e) x*y >= 1; x >= 0; y >= 0
# cannot multiply
# solution: x >= 1/y ; x affine (and concave) , 1/y for positive y is convex

problem = cp.Problem(cp.Minimize(x), [x >= cp.inv_pos(y)])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)

# (f) ( x + y )^2 / sqrt( y ) <= x - y + 5
# we cannot divide
# solution: compose quad_over_lin(u,v) with (x+y, sqrt(y)) qol is increasing in u and decreasing in v => composition with convex and concave gives convex function

problem = cp.Problem(cp.Minimize(x), [cp.quad_over_lin(x+y, cp.sqrt(y)) <= x - y + 5])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)


# (g) x^3 + y^3 <= 1; x>=0; y>=0

problem = cp.Problem(cp.Minimize(x), [x**3 + y**3 <= 1])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)


# (h) x+z <= 1+sqrt(x*y-z^2); x>=0; y>=0
# product not allowed
# equivalent to: (divide by sqrt(y) in both sides) sqrt(quad_over_lin(x,y)) + sqrt(quad_over_lin(z,y)) <= sqrt(inv_pos(y)) + sqrt(x - quad_over_lin(z,y))

# WRONG ANSWER

# from solution:

problem = cp.Problem(cp.Minimize(x), [x + z <= 1 + cp.geo_mean(cp.vstack([y, x - cp.quad_over_lin(z,y)]))])

problem.solve()

print("status:", problem.status)
print("optimal value", problem.value)
print("optimal x", x.value)
print("optimal y", y.value)

