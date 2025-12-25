import yaml
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn


class RCSSphere:
    def __init__(self, diameter):
        self.radius = diameter / 2.0
        self.c = 3e8

    def _hn(self, n, x):
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)

    def _an(self, n, x):
        return spherical_jn(n, x) / self._hn(n, x)

    def _bn(self, n, x):
        num = x * spherical_jn(n - 1, x) - n * spherical_jn(n, x)
        den = x * self._hn(n - 1, x) - n * self._hn(n, x)
        return num / den

    def rcs(self, freq, n_max=50):
        wavelength = self.c / freq
        k = 2 * math.pi / wavelength
        x = k * self.radius

        s = 0 + 0j
        for n in range(1, n_max + 1):
            s += ((-1) ** n) * (n + 0.5) * (self._bn(n, x) - self._an(n, x))

        sigma = (wavelength ** 2 / math.pi) * abs(s) ** 2
        return sigma, wavelength


class RCSOutput:
    def __init__(self):
        self.data = []

    def add(self, freq, wavelength, rcs):
        self.data.append({
            "freq": freq,
            "lambda": wavelength,
            "rcs": rcs
        })

    def save_json(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"data": self.data}, f, indent=4)

    def plot(self):
        plt.figure()
        plt.plot(
            [d["freq"] for d in self.data],
            [d["rcs"] for d in self.data]
        )
        plt.xlabel("Частота, Гц")
        plt.ylabel("ЭПР, м²")
        plt.title("ЭПР идеально проводящей сферы")
        plt.grid(True)
        plt.show()


def main():
    with open("task_rcs_02.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)["data"]

    variant = int(input("Вариант №: "))
    params = yaml_data[variant]

    D = float(params["D"])
    fmin = float(params["fmin"])
    fmax = float(params["fmax"])

    freqs = np.linspace(fmin, fmax, 150)

    sphere = RCSSphere(D)
    output = RCSOutput()

    for f in freqs:
        sigma, wavelength = sphere.rcs(f)
        output.add(float(f), float(wavelength), float(sigma))

    output.save_json("rcs_result.json")

    output.plot()


if __name__ == "__main__":
    main()
