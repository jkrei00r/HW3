# hw3b.py
import math


def t_distribution_pdf(x, m):
    """
    Probability density function of the t-distribution.

    :param x: The value at which to evaluate the PDF.
    :param m: Degrees of freedom.
    :return: The value of the PDF at x.
    """
    from math import gamma, sqrt, pi
    return (gamma((m + 1) / 2) / (sqrt(m * pi) * gamma(m / 2))) * (1 + (x ** 2) / m) ** (-(m + 1) / 2)


def gamma_function(alpha):
    """
    Compute the gamma function using a numerical approximation.

    Args:
        alpha (float): The argument to the gamma function.

    Returns:
        float: The value of the gamma function at alpha.
    """
    if alpha == 1:
        return 1.0
    elif alpha == 0.5:
        return math.sqrt(math.pi)
    else:
        # Use the recursive property of the gamma function
        return (alpha - 1) * gamma_function(alpha - 1)


def compute_K_m(m):
    """
    Compute the normalization constant K_m for the t-distribution.

    Args:
        m (int): Degrees of freedom.

    Returns:
        float: The value of K_m.
    """
    numerator = gamma_function((m + 1) / 2)
    denominator = math.sqrt(m * math.pi) * gamma_function(m / 2)
    return numerator / denominator


def trapezoidal_integration(func, a, b, *args, n=1000):
    """
    Perform numerical integration using the trapezoidal rule.

    :param func: The function to integrate.
    :param a: The lower limit of integration.
    :param b: The upper limit of integration.
    :param args: Additional arguments to pass to the function.
    :param n: The number of trapezoids to use (default is 1000).
    :return: The approximate integral of the function from a to b.
    """
    h = (b - a) / n
    integral = 0.5 * (func(a, *args) + func(b, *args))

    for i in range(1, n):
        integral += func(a + i * h, *args)

    integral *= h
    return integral


def t_distribution_cdf(z, m):
    """
    Compute the cumulative distribution function (CDF) of the t-distribution.

    :param z: The value at which to evaluate the CDF.
    :param m: Degrees of freedom.
    :return: The CDF value at z.
    """
    return trapezoidal_integration(t_distribution_pdf, -1000, z, m)


def main():
    """
    Main function to compute the CDF of the t-distribution for user-specified
    degrees of freedom and z values.
    """
    print("This program computes the CDF of the t-distribution for given degrees of freedom and z values.")

    # Test for degrees of freedom m = 7, 11, 15
    degrees_of_freedom = [7, 11, 15]

    for m in degrees_of_freedom:
        print(f"\nDegrees of freedom (m): {m}")

        # Prompt the user to input z values
        z_values = []
        for i in range(3):
            z = float(input(f"Enter z value {i + 1}: "))
            z_values.append(z)

        # Compute and print the CDF for each z value
        for z in z_values:
            cdf_value = t_distribution_cdf(z, m)
            print(f"F(z={z:0.3f} | m={m}) = {cdf_value:0.6f}")


if __name__ == "__main__":
    main()