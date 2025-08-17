from traintest import plot_formation_energies_new
from saveload import load_results_from_file
from utils import stringlist_list_converter


results1 = load_results_from_file('logs/512test/Gamma_result/result_1.txt')[0]
results4 = stringlist_list_converter(load_results_from_file('logs/512test/Gamma_result/result_4.txt'))
results5 = stringlist_list_converter(load_results_from_file('logs/512test/Gamma_result/result_5.txt'))

print(results1)
print(results4)
print(results5)

plot_formation_energies_new(results1, 'genfiles512', results4, results5)


#plot_formation_energies(list4, list5, filepath="plots/test/Gamma.png")