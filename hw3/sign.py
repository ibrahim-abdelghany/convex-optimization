from typing import Tuple

from cvxpy.atoms.atom import Atom


class sign(Atom):
    """Sign of an expression (-1 for x <= 0, +1 for x > 0).
    """
    def __init__(self, x) -> None:
        super(sign, self).__init__(x)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sign of x.
        """
        x = values[0].copy()
        x[x > 0] = 1.0
        x[x <= 0] = -1.0
        return x

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_quasiconvex(self) -> bool:
        """Is the atom quasiconvex?
        """
        return self.args[0].is_scalar()

    def is_atom_quasiconcave(self) -> bool:
        """Is the atom quasiconvex?
        """
        return self.args[0].is_scalar()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_scalar()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values) -> None:
        return None
