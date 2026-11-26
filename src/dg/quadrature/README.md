`quadrature`
================================================================================

This subdirectory contains the source code regarding numerical quadrature on the reference interval $[-1, 1]$ and the Lagrange polynomials that interpolate the quadrature nodes. These are the one-dimensional building blocks of the spectral bases of Section 2.2 of the publication (Eqs. 5a-5c): Legendre-Gauss-Lobatto (LGL) nodes in $x$ and $y$, and Legendre-Gauss (LG) nodes in $\theta$.

How It Fits Into the Whole
--------------------------------------------------------------------------------

`quadrature` depends only on `numpy`. It is used throughout the package:

- `quad_xyth` supplies the nodes and weights for every element: matrix and vector assembly in `rt` (`mass_matrix`, `scattering_matrix`, `preconditioner_matrix`, `interior_convection_matrix`, `boundary_convection_matrix`, `forcing_vector`, `boundary_conditions_vector`, `get_Ex`, `get_Ey`, `get_Eth`), `dg.projection` (`Projection`, `get_f2f_matrix`, `cell_intg_xy`, `col_intg_th`), the error indicators and *hp*-steering in `amr.error_indicator`, and the plotting routines in `tools.dg.projection`.
- `lag_eval` evaluates basis functions $\phi_i$, $\psi_j$, $\xi_a$ at points that are not nodes of the element, e.g., when coupling neighboring elements of different size or degree (`rt.get_Ex`, `rt.get_Ey`, `rt.get_Eth`, `dg.projection.get_f2f_matrix`), for Dirac boundary data (`rt.boundary_conditions_vector`), and when comparing against finer or analytic solutions (`amr.error_indicator.error`).
- `lag_ddx_eval` supplies the derivatives of the basis functions for the interior convection matrix (`rt.interior_convection_matrix`, the $\vec{s} \cdot \nabla_{\vec{x}} v$ term of Eq. 11a).
- `lg_quad` and `lgl_quad` are called only by `quad_xyth`. `lgr_quad`, `uni_quad`, `lag_ddx`, `calc_proj_mtx_1d` and `proj_1d` are exported but not used elsewhere in `src`.

Conventions
--------------------------------------------------------------------------------

- All rules here (except `lg_quad` with a non-default `intv`) live on the reference interval $[-1, 1]$. An element $[z_0, z_f]$ uses the nodes $z = z_0 + \frac{z_f - z_0}{2}(\hat{z} + 1)$ (`dg.projection.push_forward`) and the weights scaled by $\frac{z_f - z_0}{2}$ (SI-8, Eqs. 12, 15). The push-forward map in SI-8, Eq. 12 is printed with $-z_0^E$ in place of $+z_0^E$; the code adds $z_0$.
- Nodes are returned as 1-D `numpy.ndarray`s of shape `(n,)`. `quad_xyth` returns them in increasing order; `lg_quad` and `lgl_quad` return them in decreasing order; `lgr_quad` and `uni_quad` return them in increasing order.
- Exactness for $n$ nodes: LG integrates polynomials of degree $\le 2n - 1$ exactly, Legendre-Gauss-Radau (LGR) degree $\le 2n - 2$, and LGL degree $\le 2n - 3$. SI-8 states that both the LGL and LG rules with $N$ nodes are exact up to degree $2N - 1$.
- The LG nodes do not include the endpoints $\pm 1$, so angular basis functions are not interpolated at element boundaries; the LGL nodes include both endpoints and the LGR nodes include $-1$.
- `lag_eval` and `lag_ddx_eval` take a single scalar point `x`; the Lagrange basis on `nodes` is $\ell_i(x) = \prod_{j \ne i} \frac{x - x_j}{x_i - x_j}$.

Functions
--------------------------------------------------------------------------------

All functions below are exported by `__init__.py`, i.e., `from dg.quadrature import quad_xyth, lag_eval, ...`. `__init__.py` also defines `src_dir` (`str`, the path of `src`) and appends it to `sys.path` if absent.

### `quad_xyth(nnodes_x = 1, nnodes_y = 1, nnodes_th = 1) -> list`

Defined in `quad_xyth.py`. Returns the reference-interval quadrature nodes and weights for one spatial-angular element: LGL in $x$ and $y$, LG in $\theta$ (Eqs. 5a-5c).

Arguments:
- `nnodes_x` (`int`, default `1`) : Number of nodes in $x$, $n_x \ge 1$.
- `nnodes_y` (`int`, default `1`) : Number of nodes in $y$, $n_y \ge 1$.
- `nnodes_th` (`int`, default `1`) : Number of nodes in $\theta$, $n_\theta \ge 1$.

Returns `[nodes_x, weights_x, nodes_y, weights_y, nodes_th, weights_th]`, six `numpy.ndarray`s of shapes `(nnodes_x,)`, `(nnodes_x,)`, `(nnodes_y,)`, `(nnodes_y,)`, `(nnodes_th,)`, `(nnodes_th,)`, with nodes in increasing order on $[-1, 1]$ and weights summing to 2.

Notes:
- Raises `ValueError` if any argument is less than 1.
- A count of 1 returns the one-point rule `[0]`, `[2]` (the midpoint rule, which is the one-node LG rule; an LGL rule needs at least two nodes).
- Results are cached in the module-level dictionaries `lgl_nodes`, `lgl_weights` (shared by $x$ and $y$) and `lg_nodes`, `lg_weights`, keyed by the number of nodes. Repeated calls return the *same* array objects (and `nodes_x is nodes_y` when `nnodes_x == nnodes_y`), so callers must not modify the returned arrays in place. The dictionaries `lgr_nodes`, `lgr_weights`, `uni_nodes`, `uni_weights` are defined but unused.
- The symmetric LGL nodes are computed from $\cos$ and may be off from exact values by round-off (e.g., the middle node for `nnodes_x = 3` is about $1.5 \times 10^{-17}$, not $0$).

Example:

```python
from dg.quadrature import quad_xyth
from dg.projection import push_forward

[xxb, wx, yyb, wy, thb, wth] = quad_xyth(nnodes_x = 3, nnodes_y = 3, nnodes_th = 4)
# xxb = [-1, 0, 1], wx = [1/3, 4/3, 1/3]; thb are the 4 LG nodes
thf = push_forward(0., 0.5 * 3.14159, thb) # Nodes of the angular element [0, pi/2]
```

### `lg_quad(nnodes, intv = [-1, 1]) -> list`

Defined in `lg_quad.py`. Computes the Legendre-Gauss nodes (the zeros of the Legendre polynomial $P_n$) and weights by Newton iteration from a Chebyshev-based initial guess, following G. von Winckel's `lgwt` (2004) and Canuto et al., *Spectral Methods in Fluid Dynamics*, Section 2.3.

Arguments:
- `nnodes` (`int`) : Number of nodes $n \ge 1$.
- `intv` (`list` of two `float`, default `[-1, 1]`) : The interval $[a, b]$ to which nodes and weights are mapped.

Returns `[nodes, weights]`, two `numpy.ndarray`s of shape `(nnodes,)`, with nodes in *decreasing* order in $(a, b)$ and weights summing to $b - a$. Exact for polynomials of degree $\le 2n - 1$.

### `lgl_quad(nnodes = 5) -> list`

Defined in `lgl_quad.py`. Computes the Legendre-Gauss-Lobatto nodes (the endpoints $\pm 1$ and the zeros of $P'_{n-1}$) and weights $w_i = \frac{2}{n(n-1) P_{n-1}(x_i)^2}$ by Newton iteration from the Chebyshev-Gauss-Lobatto nodes, following G. von Winckel's `lglnodes` (2004).

Arguments:
- `nnodes` (`int`, default `5`) : Number of nodes $n \ge 2$. With `nnodes = 1` it divides by zero (`quad_xyth` handles one node separately).

Returns `[nodes, weights]`, two `numpy.ndarray`s of shape `(nnodes,)`, with nodes in *decreasing* order from $1$ to $-1$ and weights summing to 2. Exact for polynomials of degree $\le 2n - 3$. The docstring mentions a Vandermonde matrix, but only nodes and weights are returned.

### `lgr_quad(nnodes = 5) -> list`

Defined in `lgr_quad.py`. Computes the Legendre-Gauss-Radau nodes (the zeros of $P_{n-1} + P_n$, which include $-1$) and weights by Newton iteration from the Chebyshev-Gauss-Radau nodes, following G. von Winckel's `lgrnodes` (2004). The docstring writes the defining polynomial as "P_N + P"_N"; the code uses $P_{n-1} + P_n$.

Arguments:
- `nnodes` (`int`, default `5`) : Number of nodes $n \ge 1$.

Returns `[nodes, weights]`, two `numpy.ndarray`s of shape `(nnodes,)`, with `nodes[0] = -1`, nodes in increasing order in $[-1, 1)$, `weights[0] = 2 / nnodes**2`, and weights summing to 2. Exact for polynomials of degree $\le 2n - 2$. Not used elsewhere in `src`.

### `uni_quad(nnodes) -> list`

Defined in `uni_quad.py`. Returns `[nodes, weights]` with `nodes = numpy.linspace(-1, 1, nnodes)` and `weights = 2 * numpy.ones(nnodes)`. Intended only for comparison against other rules, and not used elsewhere in `src`. The weights sum to $2 n$, not 2, so `weights @ f(nodes)` is not an approximation of $\int_{-1}^{1} f$ (a uniform rule would need weights summing to 2, e.g., $2/n$).

Arguments:
- `nnodes` (`int`) : Number of equally spaced nodes, including both endpoints for `nnodes >= 2`.

### `lag_eval(nodes, i, x) -> float`

Defined in `lag_eval.py`. Evaluates the $i$-th Lagrange polynomial $\ell_i$ interpolating `nodes` at the point `x`.

Arguments:
- `nodes` (`numpy.ndarray`, shape `(n,)`) : Distinct interpolation nodes on the reference interval.
- `i` (`int`) : Index of the basis function, $0 \le i < n$.
- `x` (`float`) : A single evaluation point. Arrays are not supported (the range test uses Python `and` and raises `ValueError` for arrays of more than one element).

Returns $\ell_i(x)$ if $-1 - 10^{-14} \le x \le 1 + 10^{-14}$, and `0` otherwise. The zero outside the reference interval implements the convention of SI-8 that basis functions vanish outside their element, so callers can pass pulled-back coordinates of points that may lie in a neighboring element. Points that should lie on an element boundary but are pulled back slightly outside $[-1, 1]$ by round-off beyond the $10^{-14}$ tolerance evaluate to 0.

Example:

```python
import numpy as np
from dg.quadrature import lag_eval
lag_eval(np.array([-1., 0., 1.]), 1, 0.5)  # 0.75
lag_eval(np.array([-1., 0., 1.]), 1, 1.5)  # 0 (outside [-1, 1])
```

### `lag_ddx(nodes) -> numpy.ndarray`

Defined in `lag_ddx.py`. Returns the differentiation matrix `dmat` of shape `(n, n)` with `dmat[j, i]` $= \ell_i'(x_j)$, so that `dmat @ f(nodes)` approximates $f'$ at the nodes.

Arguments:
- `nodes` (`numpy.ndarray`, shape `(n,)`) : Distinct interpolation nodes.

Example: for `nodes = [-1, 0, 1]`, `lag_ddx(nodes) @ nodes**2` returns `[-2, 0, 2]`.

### `lag_ddx_eval(nodes, j, x) -> float`

Defined in `lag_ddx.py`. Evaluates the exact derivative of the $j$-th Lagrange polynomial at a scalar point, $\ell_j'(x) = \sum_{k \ne j} \frac{1}{x_j - x_k} \prod_{m \ne j, k} \frac{x - x_m}{x_j - x_m}$.

Arguments:
- `nodes` (`numpy.ndarray`, shape `(n,)`) : Distinct interpolation nodes.
- `j` (`int`) : Index of the basis function, $0 \le j < n$.
- `x` (`float`) : Evaluation point. Unlike `lag_eval`, there is no range check: the polynomial's derivative is returned for any `x`.

Returns `0` when `nodes` has a single entry. `rt.interior_convection_matrix` calls it as `lag_ddx_eval(xxb, pp, xxb[ii])`, i.e., $\phi_p'(x_i)$ on the reference interval.

`lag_ddx.py` also defines `lag_ddx_eval_approx(nodes, j, x)`, which is not exported and not used: a central finite difference of `lag_eval` with step $10^{-10}$, with the stencil clipped to $[-1, 1]$.

### `calc_proj_mtx_1d(src_nodes, trgt_nodes) -> numpy.ndarray`

Defined in `calc_proj_mtx_1d.py`. Returns the matrix `P` that maps nodal values on `src_nodes` to nodal values on `trgt_nodes` (both on $[-1, 1]$).

Arguments:
- `src_nodes` (`numpy.ndarray`, shape `(n_src,)`) : Nodes at which the input values are given.
- `trgt_nodes` (`numpy.ndarray`, shape `(n_trgt,)`) : Nodes at which output values are wanted.

Returns a `numpy.ndarray` of shape `(n_trgt, n_src)`:
- If `n_src <= n_trgt`: the interpolation matrix `P[j, i]` $= \ell^{src}_i(x^{trgt}_j)$ (evaluates the source interpolant at the target nodes).
- If `n_src > n_trgt`: the left pseudo-inverse $(A^T A)^{-1} A^T$ of `A[i, j]` $= \ell^{trgt}_j(x^{src}_i)$, i.e., the unweighted least-squares fit of a degree $n_{trgt} - 1$ polynomial to the source values.

Neither branch uses quadrature weights, so this is interpolation or discrete least squares at the nodes rather than an $L^2$ projection. In the interpolation branch, target nodes outside $[-1, 1]$ give zero rows because `lag_eval` returns 0 there. Not used elsewhere in `src` except by `proj_1d`.

### `proj_1d(func, src_nodes, trgt_nodes) -> numpy.ndarray`

Defined in `proj_1d.py`. Evaluates `func(src_nodes)` and returns `calc_proj_mtx_1d(src_nodes, trgt_nodes) @ func(src_nodes)`, the nodal values on `trgt_nodes`, of shape `(n_trgt,)`.

Arguments:
- `func` (`Callable`) : A vectorized function of one variable.
- `src_nodes`, `trgt_nodes` (`numpy.ndarray`) : As in `calc_proj_mtx_1d`.

Not used elsewhere in `src`.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `quadrature` subdirectory are:

- `__init__.py` : Adds `src` to `sys.path` and exports the functions listed above.
- `calc_proj_mtx_1d.py` : `calc_proj_mtx_1d`, the matrix mapping nodal values on one set of nodes to another (interpolation or least squares).
- `lag_ddx.py` : `lag_ddx`, `lag_ddx_eval` (and the unexported `lag_ddx_eval_approx`), for differentiating Lagrange polynomials.
- `lag_eval.py` : `lag_eval`, for evaluating Lagrange polynomials.
- `lg_quad.py` : `lg_quad`, the Legendre-Gauss rule.
- `lgl_quad.py` : `lgl_quad`, the Legendre-Gauss-Lobatto rule.
- `lgr_quad.py` : `lgr_quad`, the Legendre-Gauss-Radau rule.
- `proj_1d.py` : `proj_1d`, applies `calc_proj_mtx_1d` to a function.
- `quad_xyth.py` : `quad_xyth`, the cached nodes and weights for the $x$-, $y$-, and $\theta$-dimensions of an element.
- `uni_quad.py` : `uni_quad`, a uniform rule for comparison only.
