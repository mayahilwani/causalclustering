from ccWrapper import CCWrapper
from slope import Slope
import sys
from data_gen.CausalClustersData import generate_data

def print_usage():
    print("Usage: python script.py <n> <orig> <intv> <intv_type> <f> <clusters> <k> <r>")
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
    print("  <clusters>   : Number of clusters for data generation (integer, e.g., 2/3):")
    print("  <k>          : Number of clusters for method (integer, e.g., 2/3/4):")
    print("  <r>          : Flag for random (0 or 1):")
    print("  <mdl>          : Flag for mdl threshold break (0 or 1):")
    print("  <intv_s>     : strenght of intervention (integer, 0 to 5):")
    print("                 0: Weak Intervention")
    print("                 1: Medium Intervention")
    print("                 2: Strong Intervention .. etc")
    print("  <noise>      : Noise level (integer, 0 to 2):")
    print("                 0: Low noise")
    print("                 1: Medium noise")
    print("                 2: High noise")
    #"test_{n}_{orig}_{intv}_{intv_type}_{f}_{clusters}_{k}_{r}_{mdl_th}_{intv_strength}_{noise_level}"
   # python script.py 50 2000 500 1 0 2 3 0 0 0 0    13:04  15:16

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 12:
        print_usage()
        sys.exit(1)

    # Parse command-line arguments
    try:
        n = int(sys.argv[1])          # Number of tests
        orig = int(sys.argv[2])       # Number of original datapoints
        intv = int(sys.argv[3])       # Number of intervention datapoints
        intv_type = int(sys.argv[4])  # Type of intervention (0 to 4)
        f = int(sys.argv[5])          # Function type (0 to 2)
        clusters = int(sys.argv[6])   # Number of clusters for data generation
        k = int(sys.argv[7])          # Number of clusters for method
        r = int(sys.argv[8])          # Random flag
        mdl = int(sys.argv[9])  # Random flag
        intv_strength = int(sys.argv[10])
        noise_level = int(sys.argv[11])

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

    if clusters < 2 or clusters > 3:
        print(f"Error: clusters must be 2 or 3. Got {clusters}.")
        print_usage()
        sys.exit(1)

    if k < 2 or k > 5:
        print(f"Error: k must be between 2 and 5. Got {k}.")
        print_usage()
        sys.exit(1)

    if r < 0 or r > 1:
        print(f"Error: r must be 0 or 1. Got {r}.")
        print_usage()
        sys.exit(1)

    if mdl < 0 or mdl > 1:
        print(f"Error: mdl must be 0 or 1. Got {mdl}.")
        print_usage()
        sys.exit(1)

    if intv_strength < 0 or intv_strength > 2:
        print(f"Error: intv_strength must be between 0 and 5. Got {intv_strength}.")
        print_usage()
        sys.exit(1)

    if noise_level < 0 or noise_level > 2:
        print(f"Error: noise_level must be between 0 and 2. Got {noise_level}.")
        print_usage()
        sys.exit(1)

    # Print the arguments and their values
    print("Running script with the following arguments:")
    print(f"  n (Number of tests): {n}")
    print(f"  orig (Original datapoints): {orig}")
    print(f"  intv (Intervention datapoints): {intv}")
    print(f"  intv_type (Intervention type): {intv_type}")
    print(f"  f (Function type): {f}")
    print(f"  clusters (Number of clusters for data generation): {clusters}")
    print(f"  k (Number of clusters for method): {k}")
    print(f"  r (Random flag): {r}")
    print(f"  mdl (mdl threshold flag): {mdl}")
    print(f"  intv_strength (Intervention strength): {intv_strength}")
    print(f"  noise_level (Noise level): {noise_level}")

    # Filename
    foldername = f"test_{n}_{orig}_{intv}_{intv_type}_{f}_{clusters}_{k}_{r}_{mdl}_{intv_strength}_{noise_level}"

    # Data gen filepath
    dg_filepath = f"./tests/{foldername}"

    # Filepath
    filepath = f"./tests/{foldername}"

    # Generate the test data
    generate_data(dg_filepath, n, orig, intv, intv_type, f, clusters, intv_strength, noise_level)

    # Initialize SpotWrapper
    cc = CCWrapper()
    cc.generate_stats(filepath, n, k, [], r, mdl)


if __name__ == "__main__":
    main()