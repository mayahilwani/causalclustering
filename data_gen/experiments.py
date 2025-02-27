import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, logistic

# Set the seed for reproducibility
np.random.seed(42)


def generate_plots(num_samples=100, num_plots=4):
    x = np.linspace(-10, 10, num_samples)

    fig, axs = plt.subplots(num_plots, num_plots, figsize=(20, 20))
    fig.suptitle("Relationships between Two Variables", fontsize=20)

    for i in range(num_plots):
        for j in range(num_plots):
            ax = axs[i, j]

            if i == 0:
                # Linear with Gaussian noise
                y = 2 * x + np.random.normal(0, 5, num_samples)
                ax.set_title("Linear with Gaussian Noise")

            elif i == 1:
                # Linear with Non-Gaussian noise (e.g., Laplace)
                y = 2 * x + laplace.rvs(scale=5, size=num_samples)
                ax.set_title("Linear with Non-Gaussian Noise (Laplace)")

            elif i == 2:
                # Nonlinear with Gaussian noise
                y = np.sin(x) + np.random.normal(0, 0.5, num_samples)
                ax.set_title("Nonlinear (Sin) with Gaussian Noise")

            elif i == 3:
                # Nonlinear with Non-Gaussian noise (e.g., Logistic)
                y = np.sin(x) + logistic.rvs(scale=0.5, size=num_samples)
                ax.set_title("Nonlinear (Sin) with Non-Gaussian Noise (Logistic)")

            # Plot the generated data
            ax.scatter(x, y, color='blue', alpha=0.7)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Generate the plots
generate_plots()
