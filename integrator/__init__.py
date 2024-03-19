from .el import Euler, ImplicitEuler
from .rk import RK4

__factory = {
    "RK4": RK4,
    "Euler": Euler,
    "ImplicitEuler": ImplicitEuler,
}


def ODEIntegrate(method, *args, **kwargs):
    """
    Solves an ODE using a numerical integration method.

    Args:
        func (callable): The ODE function to be solved.
        t0 (float): The initial time.
        t1 (float): The final time.
        dt (float): The time step for integration.
        y0 (torch.Tensor): The initial state of the system.
        device (torch.device, optional): The device to use for computation (default: None).
        dtype (torch.dtype, optional): The data type to use for computation (default: None).
        *args: Additional positional arguments to be passed to the ODE function.
        **kwargs: Additional keyword arguments to be passed to the ODE function.

    Return:
        (bs, num_timesteps, num_states)

    Raises:
            Exception: If NaN values are encountered during integration.
    """
    if method not in __factory.keys():
        raise ValueError("solver '{}' is not implemented".format(method))
    results = __factory[method](*args, **kwargs).solve(*args, **kwargs)
    return results
