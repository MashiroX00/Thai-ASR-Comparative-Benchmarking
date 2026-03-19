import csv
import matplotlib.pyplot as plt
import datetime
def readCsv():
    try:
        with open('asr_benchmark_results.csv', 'r') as file:
            data = csv.DictReader(file)
            data = list(data)
            return data
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    exit()

def draw_graphs(data, col, save=False):
    plt.figure(figsize=(10, 5))
    plt.title(col)
    plt.xlabel("Model")
    plt.ylabel(col)
    models = []
    y_axis = []
    for row in data:
        models.append(row['Model'])
        y_axis.append(float(row[col]))
    plt.bar(models, y_axis)
    if save:
        plt.savefig(f"{col}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()
data = readCsv()
col_names = []
for i in data[0]:
    col_names.append(i)

# GUI Drawing 2-6
while True:
    for i in range(2, len(col_names)):
        print(f"[{i - 1}] : {col_names[i]}")
    print(f"[0] : Exit")
    try:
        choice = int(input(f"Enter number to draw graph: ")) + 1
    except ValueError:
        print("Invalid input. Please enter a number.")
        break
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    if choice != 1 and choice < len(col_names):
        is_save = input(f"Save {col_names[choice]} graph? (y/n): ").startswith('y')
        draw_graphs(data, col_names[choice], is_save)
    else:
        break