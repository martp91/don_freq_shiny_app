import numpy as np
from scipy.integrate import odeint

def blood_volume_func(height, weight, sex):
    "height in m, weight in kg"
    if sex.lower() == "male":
        return 0.3669 * height**3 + 0.03219 * weight + 0.6041
    else:
        return 0.3561 * height**3 + 0.03308 * weight + 0.1833


CONVERT_MMOL = 0.0621
MU_LF = 1.7  # fer=50ng/mL
S_LF = 0.37

# # from cable 2016 et al?
# # rescale to reduce correlations
A = -2.05
B = 15.6
C = -13.9
Ap = A * S_LF**2
Bp = B * S_LF + 2 * Ap * MU_LF / S_LF
Cp = C + Bp * MU_LF / S_LF - Ap * MU_LF**2 / S_LF**2


def iron_log_fer_func(log_fer, a, b, c):
    log_fer_ = (log_fer - MU_LF) / S_LF
    iron = c + b * log_fer_ + a * log_fer_**2
    return iron


def inv_iron_log_fer_func(iron, a, b, c):
    D = b**2 - 4 * a * (c - iron)
    D = np.where(D < 0, 0, D)
    log_fer_ = (-b + D**0.5) / (2 * a)
    return S_LF * log_fer_ + MU_LF


def ode_model(y, t, Hb_base, fer_base, alpha, beta, gamma, kappa):
    Hb, fer = y
    dHb_dt = alpha * np.exp(kappa * (fer / 728 - 1)) * (Hb_base - Hb)
    dfer_dt = beta * (fer_base - fer) - gamma * dHb_dt
    return dHb_dt, dfer_dt


def Hb_fer_model_iron(
    don_times,
    taken_vol,
    V,
    BW,
    Hb_base,
    fer_base,
    alpha,
    beta,
    gamma,
    kappa,
    iron_a,
    iron_b,
    iron_c,
    loss_scale=1.0,
):

    Hb_base *= V / CONVERT_MMOL * 3.38  # iron in Hb mg/g
    f_losses = 1 - taken_vol / V * loss_scale
    dts = don_times[1:] - don_times[:-1]

    iron_base = iron_log_fer_func(np.log10(fer_base), iron_a, iron_b, iron_c) * BW

    ndons = len(don_times)
    y0 = [Hb_base, iron_base]
    out = [y0]
    for dt, f_loss in zip(dts, f_losses[:-1]):
        y0 = out[-1].copy()
        y0[0] *= f_loss
        y = odeint(
            ode_model, y0, [0, dt], args=(Hb_base, iron_base, alpha, beta, gamma, kappa)
        )
        out.append(y[-1])

    y = np.array(out)
    Hb, iron = y.T
    Hb /= V / CONVERT_MMOL * 3.38
    log_fer = inv_iron_log_fer_func(iron / BW, iron_a, iron_b, iron_c)
    fer = 10**log_fer
    return Hb, fer

