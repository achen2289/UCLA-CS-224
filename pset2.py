from collections import defaultdict
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def q3():
    with open("q3 data/gwas.geno", "r") as f:
        genos = f.read().strip()
    
    genos_clean = []
    for line in genos.split("\n"):
        geno_line = line.strip().split(" ")
        genos_clean.append([int(geno) for geno in geno_line])

    with open("q3 data/gwas.pheno", "r") as f:
        phenos = f.read().strip()
    
    phenos_clean = []
    for line in phenos.split("\n"):
        pheno_line = line.strip().split(" ")
        phenos_clean.append([float(pheno) for pheno in pheno_line])
    
    significant_datapoints = defaultdict(list) # map phenotype to genotype index for significant vals
    alpha = 0.05 / 382
    num_phenos = len(phenos_clean[0])
    num_genos = len(genos_clean[0])
    num_datapoints = len(phenos_clean)
    for i in range(num_phenos): # go through each phenotype
        curr_pheno = [pheno[i] for pheno in phenos_clean]
        SNP_p_values = []
        for j in range(num_genos): # go through each SNP
            curr_SNPs = [geno[j] for geno in genos_clean]
            curr_SNPs = sm.add_constant(curr_SNPs)
            model = sm.OLS(curr_pheno, curr_SNPs)
            results = model.fit()

            if results.pvalues[1] < alpha:
                significant_datapoints[i].append(j)
            
            SNP_p_values.append(results.pvalues[1])

        # plt.hist(SNP_p_values, bins=30, color="turquoise")
        # plt.xlabel("p values")
        # plt.ylabel("Frequency")
        # plt.title(f"p Value distribution for Phenotype {i}")
        # plt.show()

    print (significant_datapoints)

    random_noises = [random.uniform(0, 0.1) for _ in range(len(genos_clean))]
    for pheno_idx, geno_idxs in significant_datapoints.items():
        for geno_idx in geno_idxs:
            curr_phenos = [p[pheno_idx] for p in phenos_clean]
            curr_genos = [g[geno_idx] for g in genos_clean]
            genos_noise = [g + noise for g, noise in zip(curr_genos, random_noises)]
            plt.scatter(genos_noise, curr_phenos, s=0.4)
            plt.xlabel("Genotypes")
            plt.ylabel("Phenotypes")
            plt.title(f"Phenotype ({pheno_idx}) Versus Genotype ({geno_idx})")
            print (len(curr_phenos))
            if pheno_idx == 3:
                print (curr_phenos)
            plt.show()

def q4():
    with open("q4 data/ridge.test.geno", "r") as f:
        geno_test = f.read().strip()
    
    geno_test_clean = []
    for line in geno_test.split("\n"):
        geno_test_clean.append([int(g) for g in line.split()])
    
    with open("q4 data/ridge.test.pheno", "r") as f:
        pheno_test = f.read().strip()

    pheno_test_clean = []
    for line in pheno_test.split("\n"):
        pheno_test_clean.append([float(p) for p in line.split()])

    with open("q4 data/ridge.training.geno", "r") as f:
        geno_train = f.read().strip()
    
    geno_train_clean = []
    for line in geno_train.split("\n"):
        geno_train_clean.append([int(g) for g in line.split()])
    
    with open("q4 data/ridge.training.pheno", "r") as f:
        pheno_train = f.read().strip()
    
    pheno_train_clean = []
    for line in pheno_train.split("\n"):
        pheno_train_clean.append([float(p) for p in line.split()])
    
    # for data in (geno_test_clean, pheno_test_clean, geno_train_clean, pheno_train_clean):
    #     print ([x[0] for x in data[:5]])
    #     print (len(data))

    alphas = [0.001, 2, 5, 8]
    training_mses = []
    testing_mses = []
    for a in alphas:
        clf = Ridge(alpha=a)
        clf.fit(geno_train_clean, pheno_train_clean)

        pheno_train_predict = clf.predict(geno_train_clean)
        training_mse = mean_squared_error(pheno_train_clean, pheno_train_predict)
        training_mses.append(training_mse)

        pheno_test_predict = clf.predict(geno_test_clean)
        testing_mse = mean_squared_error(pheno_test_clean, pheno_test_predict)
        testing_mses.append(testing_mse)
    
    print (training_mses)
    plt.scatter(alphas, training_mses, label="Training MSEs")
    plt.scatter(alphas, testing_mses, label="Testing MSEs")
    plt.xlabel("Lambda value")
    plt.ylabel("MSEs")
    plt.title("MSE versus Lambda, on Training and Testing Data")
    plt.legend()
    plt.show()
    
    


if __name__ == "__main__":
    # q3()
    q4()