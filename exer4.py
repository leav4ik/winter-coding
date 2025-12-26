import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import os

f_hz = 0.3e9
ratio_2l_lam = 1.2
c = 3e8
lam = c / f_hz
k = 2 * np.pi / lam
l = (ratio_2l_lam * lam) / 2
kl = k * l

def calculate_field_pattern(theta_rad):
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.cos(kl * np.cos(theta_rad)) - np.cos(kl)
        denominator = np.sin(theta_rad)
        res = np.abs(numerator / denominator)
    return np.nan_to_num(res, nan=0.0)

theta_deg = np.arange(0, 361, 1)
theta_rad = np.deg2rad(theta_deg)

f_unnorm = calculate_field_pattern(theta_rad)
f_max = np.max(f_unnorm)
f_norm = f_unnorm / f_max

theta_int = np.linspace(0, np.pi, 500)
f_int = calculate_field_pattern(theta_int) / f_max
integrand = (f_int**2) * np.sin(theta_int)
integral_val = simpson(y=integrand, x=theta_int)

d_max_linear = 2 / integral_val
d_max_db = 10 * np.log10(d_max_linear)

d_theta_lin = d_max_linear * (f_norm**2)
d_theta_db = 10 * np.log10(np.clip(d_theta_lin, 1e-4, None))

sim_data = np.loadtxt('results_variant23.txt')
sim_theta = sim_data[:, 0]
sim_db = sim_data[:, 1]
sim_lin = 10**(sim_db / 10)

print(f"Максимальный КНД (Dmax) в разах:    {d_max_linear:.4f}")
print(f"Максимальный КНД (Dmax) в дБ:      {d_max_db:.2f} дБ")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Диаграмма направленности ($2l/\\lambda=1.2$)')

axs[0,0].plot(theta_deg, d_theta_lin, 'b-', label='Аналитика')
axs[0,0].plot(sim_theta, sim_lin, 'r--', label='Симуляция (CST)')
axs[0,0].set_title('КНД (разы)')
axs[0,0].grid(True); axs[0,0].legend()

axs[0,1].plot(theta_deg, d_theta_db, 'b-', label='Аналитика')
axs[0,1].plot(sim_theta, sim_db, 'r--', label='Симуляция (CST)')
axs[0,1].set_title('КНД (дБ)')
axs[0,1].set_ylim(d_max_db-30, d_max_db+5)
axs[0,1].grid(True); axs[0,1].legend()

ax3 = plt.subplot(223, projection='polar')
ax3.set_theta_zero_location("N")
ax3.plot(theta_rad, d_theta_lin, 'b-', label='Аналитика')
ax3.plot(np.deg2rad(sim_theta), sim_lin, 'r--', label='Симуляция')
ax3.set_title('КНД (разы)')

ax4 = plt.subplot(224, projection='polar')
ax4.set_theta_zero_location("N")
ax4.plot(theta_rad, d_theta_db, 'b-', label='Аналитика')
ax4.plot(np.deg2rad(sim_theta), sim_db, 'r--', label='Симуляция')
ax4.set_ylim(d_max_db-30, d_max_db+5)
ax4.set_title('КНД (дБ)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()