from traintest import load_results_from_file, plot_formation_energies_new

def stringlist_list_converter(string_list):
    # Strings zusammenfügen und nur den Zahlen-Teil extrahieren
    full_str = " ".join(string_list)
    start = full_str.find('[') + 1
    end = full_str.find(']')
    numbers_str = full_str[start:end]

    # In float-Liste umwandeln
    numbers = [float(x) for x in numbers_str.replace(',', ' ').split()]
    
    # PyTorch-Tensor erzeugen
    return numbers


results1 = load_results_from_file('logs/512test/Gamma_result/result_1.txt')[0]
results4 = stringlist_list_converter(load_results_from_file('logs/512test/Gamma_result/result_4.txt'))
results5 = stringlist_list_converter(load_results_from_file('logs/512test/Gamma_result/result_5.txt'))

print(results1)
print(results4)
print(results5)

plot_formation_energies_new(results1, 'genfiles512', results4, results5)


#plot_formation_energies(list4, list5, filepath="plots/test/Gamma.png")