import numpy as np
import glob

l_files = glob.glob("average_membrane_potentials_*")

for i_file, filename in enumerate(l_files):
    data  = np.load(filename)
    if not i_file:
        data_shape = [len(l_files)]
        for axis in data.shape:
            data_shape.append(axis)
        test_data = np.zeros(data_shape)
    test_data[i_file] = data    
print("test_data[0, :, 0]")
print(test_data[0, :, 0])
print("test_data[1, :, 0]")
print(test_data[1, :, 0])
diff = test_data[1] - test_data[0]
print("np.linalg.norm(diff)")
print(np.linalg.norm(diff))
