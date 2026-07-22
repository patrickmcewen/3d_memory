import matplotlib.pyplot as plt
import numpy as np
from math import log2
def main():
    w_pitch = 100
    l_pitch = 100
    w_dim = 26000
    l_dim = 33000
    w = w_dim / w_pitch
    l = l_dim / l_pitch
    t_access = 4
    t_sa = 2
    h = np.linspace(1, int(t_access/t_sa), int(t_access/t_sa))

    bw = w*l*(h+1)/(2*t_access)
    bw_ideal = [w*l/t_sa] * len(h)

    print(f"1 layer BW: {bw[0]}")
    print(f"Best harmonic BW: {bw[-1]} with {h[-1]} staircase tiers")
    print(f"Ideal BW: {bw_ideal[0]}")

    #actual_3d_layers = h + [sum(np.ceil(0.5*np.log2(h))[:i]) for i in range(len(h))]
    actual_3d_layers = h*3 - 1
    print(f"Actual 3D layers: {actual_3d_layers[-1]}")

    capacity = w*l*(h[-1]+1)/2
    capacity_ideal = w*l*actual_3d_layers[-1]
    print(f"Capacity: {capacity}")
    print(f"Ideal Capacity: {capacity_ideal}")

    cap_optimized_bandwidth = w*l/t_access
    print(f"Capacity Optimized BW: {cap_optimized_bandwidth}")

    print(f"bandwidth increase for bandwidth optimized config: {bw[-1]/cap_optimized_bandwidth}x")
    print(f"capacity overhead for bandwidth optimized config: {capacity_ideal/capacity}x")

    plt.plot(actual_3d_layers, bw, label='BW')
    plt.plot(actual_3d_layers, bw_ideal, label='BW Ideal')
    plt.legend()
    plt.yscale('log')
    plt.xlabel("number of 3d layers")
    plt.savefig('bw.png')

if __name__ == "__main__":
    main()