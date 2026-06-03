import numpy as np
from scipy.integrate import odeint


def blood_volume_func(height, weight, sex):
    """Estimate donor blood volume in liters.
    Nadlers formula

    Args:
        height: Donor height in meters.
        weight: Donor weight in kilograms.
        sex: Biological sex label ("male" or anything else treated as female).

    Returns:
        Estimated blood volume in liters.
    """
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
    """Map log10 ferritin to estimated iron mass contribution.

    The ferritin term is standardized using model constants before applying
    a quadratic transform.

    From Cable 2016
    Args:
        log_fer: Base-10 logarithm of ferritin concentration.
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Intercept.

    Returns:
        Iron estimate on the model scale.
    """
    log_fer_ = (log_fer - MU_LF) / S_LF
    iron = c + b * log_fer_ + a * log_fer_**2
    return iron


def inv_iron_log_fer_func(iron, a, b, c):
    """Invert the quadratic ferritin-to-iron transform.
    inverse of iron_log_fer_func, solving for log_fer given iron on the model scale
    Args:
        iron: Iron value on the model scale.
        a: Quadratic coefficient used in the forward transform.
        b: Linear coefficient used in the forward transform.
        c: Intercept used in the forward transform.

    Returns:
        Base-10 logarithm of ferritin concentration.
    """
    D = b**2 - 4 * a * (c - iron)
    D = np.where(D < 0, 0, D) # handle small negative values due to numerical issues
    log_fer_ = (-b + D**0.5) / (2 * a)
    return S_LF * log_fer_ + MU_LF


def ode_model(y, t, Hb_base, fer_base, alpha, beta, gamma, kappa):
    """Compute time derivatives for hemoglobin and iron stores.

    Args:
        y: Current state vector ``(Hb, fer)`` in model units.
        t: Time value passed by ``odeint`` (unused directly).
        Hb_base: Baseline hemoglobin equilibrium target.
        fer_base: Baseline iron/ferritin equilibrium target.
        alpha: Hemoglobin recovery rate parameter.
        beta: Ferritin recovery rate parameter.
        gamma: Coupling between hemoglobin synthesis and iron depletion.
        kappa: Nonlinear sensitivity of Hb recovery to ferritin status.

    Returns:
        Tuple ``(dHb_dt, dfer_dt)`` for ODE integration.
    """
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
    """Simulate hemoglobin and ferritin trajectories across donations.

    The model applies an immediate fractional blood loss at each donation,
    then integrates recovery dynamics until the next donation event.

    Args:
        don_times: Donation timestamps in days.
        taken_vol: Fraction (or effective fraction) of blood volume removed
            per donation timestamp.
        V: Blood volume in liters.
        BW: Body weight in kilograms.
        Hb_base: Baseline hemoglobin in mmol/L.
        fer_base: Baseline ferritin in ng/mL.
        alpha: Hemoglobin recovery rate parameter.
        beta: Iron/ferritin recovery rate parameter.
        gamma: Coupling between Hb production and iron use.
        kappa: Ferritin sensitivity exponent for Hb recovery.
        iron_a: Quadratic coefficient for ferritin-to-iron transform.
        iron_b: Linear coefficient for ferritin-to-iron transform.
        iron_c: Intercept for ferritin-to-iron transform.
        loss_scale: Optional multiplier for donation loss severity.

    Returns:
        Tuple ``(Hb, fer)`` arrays evaluated at donation timestamps.
    """

    #convert from Hb in mmol/L and iron in mg/kg to model units
    Hb_base *= V / CONVERT_MMOL * 3.38  # iron in Hb mg/g
    f_losses = 1 - taken_vol / V * loss_scale
    dts = don_times[1:] - don_times[:-1]

    iron_base = iron_log_fer_func(np.log10(fer_base), iron_a, iron_b, iron_c) * BW

    y0 = [Hb_base, iron_base]
    out = [y0]
    #loop over donations and apply loss then integrate until next donation
    #take input of previous to next
    for dt, f_loss in zip(dts, f_losses[:-1]):
        y0 = out[-1].copy()
        y0[0] *= f_loss
        y = odeint(
            ode_model, y0, [0, dt], args=(Hb_base, iron_base, alpha, beta, gamma, kappa)
        )
        out.append(y[-1])

    y = np.array(out)
    #deconvert from model units to Hb in mmol/L and iron in mg/kg
    Hb, iron = y.T
    Hb /= V / CONVERT_MMOL * 3.38
    log_fer = inv_iron_log_fer_func(iron / BW, iron_a, iron_b, iron_c)
    fer = 10**log_fer
    return Hb, fer
