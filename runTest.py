from spotWrapper import SpotWrapper
from slope import Slope
import sys

def print_usage():
    print("Usage: python script.py <n> <orig> <intv> <intv_type> <f>")
    print("\nArguments:")
    print("  <n>          : Number of tests (integer, e.g., 10/50/100)")
    print("  <orig>       : Number of original datapoints (integer, e.g., 2000)")
    print("  <intv>       : Number of intervention datapoints (integer, e.g., 500)")
    print("  <intv_type>  : Type of intervention (integer, 0 to 4):")
    print("                 0: Flip Intervention")
    print("                 1: Scale Intervention")
    print("                 2: Shift Intervention")
    print("                 3: Random Intervention")
    print("                 4: Intervention")
    print("  <f>          : Function type (integer, 0 to 2):")
    print("                 0: Function type Polynomial degree 1")
    print("                 1: Function type Polynomial degree 2")
    print("                 2: Function type Periodic")
    print("  <c>          : Number of clusters (integer, e.g., 2/3/4):")

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 6:
        print_usage()
        sys.exit(1)

    # Parse command-line arguments
    try:
        n = int(sys.argv[1])          # Number of tests
        orig = int(sys.argv[2])       # Number of original datapoints
        intv = int(sys.argv[3])       # Number of intervention datapoints
        intv_type = int(sys.argv[4])  # Type of intervention (0 to 4)
        f = int(sys.argv[5])          # Function type (0 to 2)
        c = int(sys.argv[6])          # Number of clusters
    except ValueError:
        print("Error: All arguments must be integers.")
        print_usage()
        sys.exit(1)

    # Validate argument ranges
    if intv_type < 0 or intv_type > 3:
        print(f"Error: intv_type must be between 0 and 3. Got {intv_type}.")
        print_usage()
        sys.exit(1)

    if f < 0 or f > 2:
        print(f"Error: f must be between 0 and 2. Got {f}.")
        print_usage()
        sys.exit(1)

    # Print the arguments and their values
    print("Running script with the following arguments:")
    print(f"  n (Number of tests): {n}")
    print(f"  orig (Original datapoints): {orig}")
    print(f"  intv (Intervention datapoints): {intv}")
    print(f"  intv_type (Intervention type): {intv_type}")
    print(f"  f (Function type): {f}")
    print(f"  c (Function type): {c}")

    # Filename
    foldername = f"test_{n}_{orig}_{intv}_{intv_type}_{f}_{c}"

    # Filepath
    filepath = f"./data_gen/tests/{foldername}"

    # Generate the test data
    #TODO

    # Initialize SpotWrapper
    spot = SpotWrapper()

    # Generate stats using the provided arguments
    spot.generate_stats(filepath, n, orig, intv, intv_type, f, c)

if __name__ == "__main__":
    main()