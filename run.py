import os
import json
# os.system("python non_parametric_approx.py --CG_steps 20 --kNN 6 --PCA --dataset MNIST ")
# os.system("python non_parametric_approx.py --CG_steps 20 --kNN 6 --PCA --dataset FashionMNIST")
# os.system("python non_parametric_approx.py --CG_steps 20 --kNN 6 --PCA --dataset USPS --num_experiments 10")

# os.system("python non_parametric_approx.py --CG_steps 20 --kNN 6 --PCA --dataset MNIST --all True --step_size .001")
# os.system("python non_parametric_approx.py --CG_steps 20 --kNN 6 --PCA --dataset FashionMNIST --all True --step_size .001")
# os.system("python non_parametric_approx.py --CG_steps 20 --kNN 6 --PCA --dataset USPS --all True")
overall_list = []
overall_time = 0
for file in os.listdir("intervals/"):
    print(file)
    with open(f"intervals/{file}") as f:
        d = json.load(f)
    print(file)
    # print("avg length")
    overall_list += [end - start for start, end in d["intervals"] if end!= float('inf') and start != float('inf')]
    print(sum([end - start for start, end in d["intervals"]]) / len(d["intervals"]))
    # print("avg time")
    overall_time += d["time"]
print(overall_time / 10, len(overall_list) / 10)
print(f"time: {overall_time / len(overall_list)}")
print(sum(overall_list) / len(overall_list))
