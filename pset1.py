from concurrent import futures
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, linregress, pearsonr
import random

def main():
    with open("data/ps1.genos") as f:
        genos = f.readlines()
    
    with open("data/ps1.phenos") as f:
        phenos = f.readlines()
    
    genos_clean = []
    for line in genos:
        line = line.split(" ")
        line = [int(x) for x in line]
        genos_clean.append(line)
    
    phenos_clean = [float(x.strip()) for x in phenos]

    # print (len(genos_clean))
    # print (len(phenos_clean))

    # print (genos_clean[0])
    # print (phenos_clean[0])
    
    # data = [(x, y) for x, y in zip(genos_clean, phenos_clean)]
    # print (data[:1])
    # permutations(data[:1])
    # print (permutations([(1, 2, 3), (2, 3, 1)]))
    # print (permutations([(1), (2)]))

    
    permutation_T_vals = permutation_test(genos_clean, phenos_clean)
    observed_T_val = (pearsonr([x[0] for x in genos_clean], phenos_clean).statistic ** 2) * len(genos)
    print (permutation_T_vals)
    print (observed_T_val)
    plot_plots(permutation_T_vals, observed_T_val)

    total_greater_than_or_equal = 0
    for val in permutation_T_vals:
        if val >= observed_T_val:
            total_greater_than_or_equal += 1
    
    assert (len(permutation_T_vals) == 100000)
    p_val = total_greater_than_or_equal / len(permutation_T_vals)
    print (total_greater_than_or_equal)
    print (p_val)

def main2():
    plot_chi_squared()



def permutation_test(genos, phenos):
    T_vals = []
    for i in range(100000):
        perm_genos = permutations(genos) # generate permutations of half of data
        # perm_genos, perm_phenos = [x[0] for x in perm_data], [x[1] for x in perm_data]
        perm_genos_col_1 = [geno[0] for geno in perm_genos]
        # print (perm_phenos.index(-0.182276997996379)) == perm_genos.index([2, 0, 2, 0, 2, 2, 0, 0, 2, 2])
        T_val = (pearsonr(perm_genos_col_1, phenos).statistic ** 2) * (len(genos))
        T_vals.append(T_val)
    return T_vals
    
def permutations(data):
    return random.sample(data, len(data))

def plot_plots(permutation_T_vals, observed_T_val):
    fig, axs = plt.subplots(1, 1, layout="constrained")
    # axs[0].set_title("T1 Test Statistic Across 100,000 Permutations")
    # axs[0].set(xlabel="Iteration", ylabel="T1 value")
    # axs[0].scatter(range(1, len(permutation_T_vals)+1), permutation_T_vals)
    axs.hist(permutation_T_vals, bins=20, color="turquoise")
    axs.set_title("Histogram of T1 Test Statistic Across 100,000 Permutations")
    axs.set(xlabel="T1 value", ylabel="Count")
    plt.axvline(x=observed_T_val, c="r")
    plt.text(observed_T_val, 1.0, f"T1 = {round(observed_T_val, 2)}")
    plt.show()

def plot_chi_squared():
    x = np.linspace(0, 20, 1000)
    pdf = chi2.pdf(x, 1)
    plt.plot(x, pdf, color="turquoise")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Chi-squared Probability Density Function, 1 DF")
    plt.show()


if __name__ == "__main__":
    main()