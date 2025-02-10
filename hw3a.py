#hw3a.py
from numericalMethods import GPDF, Probability


def secant_method(func, target, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method to find the root of the equation func(x) - target = 0.

    Args:
        func (function): The function for which the root is to be found.
        target (float): The target value (desired probability).
        x0 (float): Initial guess for the root.
        x1 (float): Second initial guess for the root.
        tol (float): Tolerance for convergence (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        float: The value of x that satisfies func(x) = target within the given tolerance.
    """
    for i in range(max_iter):
        # Evaluate the function at x0 and x1
        fx0 = func(x0) - target
        fx1 = func(x1) - target

        # Check if the function value at x1 is within tolerance
        if abs(fx1) < tol:
            return x1

        # Avoid division by zero
        if fx1 - fx0 == 0:
            break

        # Update x2 using the secant formula
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        # Update x0 and x1 for the next iteration
        x0, x1 = x1, x2

    # Return the best approximation after max_iter iterations
    return x1


def main():
    """
    Main function to interactively compute probabilities or find the value of c
    for a given probability using the Gaussian Normal Distribution.

    The program allows the user to:
    1. Specify c and compute the corresponding probability.
    2. Specify a probability and find the corresponding value of c.

    The program handles both one-sided and two-sided probabilities.
    """
    # Initialize default values
    Again = True
    mean = 0
    stDev = 1.0
    c = 0.5
    OneSided = True  # Default to one-sided integration
    GT = False  # Default to P(x < c)
    yesOptions = ["y", "yes", "true"]  # Valid responses for "yes"

    while Again:
        # Ask the user whether they want to specify c and seek P or specify P and seek c
        response = input("Do you want to specify c and seek P (1) or specify P and seek c (2)? ").strip().lower()

        # Validate user input
        if response not in ["1", "2"]:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        # Case 1: User specifies c and seeks P
        if response == "1":
            # Solicit user input for mean, standard deviation, and c
            response = input(f"Population mean? ({mean:0.3f})").strip().lower()
            mean = float(response) if response != '' else mean

            response = input(f"Standard deviation? ({stDev:0.3f})").strip().lower()
            stDev = float(response) if response != '' else stDev

            response = input(f"c value? ({c:0.3f})").strip().lower()
            c = float(response) if response != '' else c

            # Ask if the probability is greater than c
            response = input(f"Probability greater than c? ({GT})").strip().lower()
            GT = True if response in yesOptions else False

            # Ask if the integration is one-sided or two-sided
            response = input(f"One sided? ({OneSided})").strip().lower()
            OneSided = True if response in yesOptions else False

            # Compute the probability based on user input
            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT=GT)
                print(f"P(x" + (
                    ">" if GT else "<") + f"{c:0.2f}" + "|" + f"{mean:0.2f}" + ", " + f"{stDev:0.2f}" + f") = {prob:0.2f}")
            else:
                prob = Probability(GPDF, (mean, stDev), c, GT=True)
                prob = 1 - 2 * prob
                if GT:
                    print(f"P({mean - (c - mean)}>x>{mean + (c - mean)}|{mean:0.2f},{stDev:0.2f}) = {1 - prob:0.3f}")
                else:
                    print(f"P({mean - (c - mean)}<x<{mean + (c - mean)}|{mean:0.2f},{stDev:0.2f}) = {prob:0.3f}")

        # Case 2: User specifies P and seeks c
        else:
            # Solicit user input for mean and standard deviation
            response = input(f"Population mean? ({mean:0.3f})").strip().lower()
            mean = float(response) if response != '' else mean

            response = input(f"Standard deviation? ({stDev:0.3f})").strip().lower()
            stDev = float(response) if response != '' else stDev

            # Ask for the desired probability
            target_prob = float(input("Enter the desired probability: ").strip())

            # Ask if the probability is greater than c
            response = input(f"Probability greater than c? ({GT})").strip().lower()
            GT = True if response in yesOptions else False

            # Ask if the integration is one-sided or two-sided
            response = input(f"One sided? ({OneSided})").strip().lower()
            OneSided = True if response in yesOptions else False

            # Define the probability function to be used in the Secant method
            def prob_func(c):
                """
                Computes the probability for a given c based on user input.
                """
                if OneSided:
                    return Probability(GPDF, (mean, stDev), c, GT=GT)
                else:
                    prob = Probability(GPDF, (mean, stDev), c, GT=True)
                    return 1 - 2 * prob

            # Initial guesses for c (can be adjusted based on the problem)
            c0 = mean - 2 * stDev
            c1 = mean + 2 * stDev

            # Use the Secant method to find the value of c that matches the desired probability
            c = secant_method(prob_func, target_prob, c0, c1)

            # Print the result based on user input
            if OneSided:
                print(f"c value for P(x" + (
                    ">" if GT else "<") + f"c|{mean:0.2f},{stDev:0.2f}) = {target_prob:0.3f} is {c:0.3f}")
            else:
                if GT:
                    print(
                        f"c value for P({mean - (c - mean)}>x>{mean + (c - mean)}|{mean:0.2f},{stDev:0.2f}) = {target_prob:0.3f} is {c:0.3f}")
                else:
                    print(
                        f"c value for P({mean - (c - mean)}<x<{mean + (c - mean)}|{mean:0.2f},{stDev:0.2f}) = {target_prob:0.3f} is {c:0.3f}")

        # Ask the user if they want to run the program again
        response = input(f"Go again? (Y/N)").strip().lower()
        Again = True if response in ["y", "yes", "true"] else False


if __name__ == "__main__":
    main()