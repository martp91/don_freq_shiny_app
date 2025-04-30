from collections import namedtuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from shiny import render, ui, reactive
from shiny.express import input
from shiny.express import ui as eui


def blood_volume_func(height, weight, sex):
    "height in m, weight in kg"
    if sex == "male":
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


DonorData = namedtuple(
    "DonorData",
    ["don_times", "taken_vol", "blood_volume", "body_weight", "Hb_base", "fer_base"],
)
ModelParams = namedtuple(
    "ModelParams",
    ["alpha", "beta", "gamma", "kappa", "iron_a", "iron_b", "iron_c"],
)


def model(donor_data, model_params):
    return Hb_fer_model_iron(*donor_data, *model_params)


MALE_FER = 100.0
MALE_HB = 9.4
FEMALE_FER = 50.0
FEMALE_HB = 8.5
MALE_WEIGHT = 85
FEMALE_WEIGHT = 73
MALE_HEIGHT = 1.85
FEMALE_HEIGHT = 1.71

MODEL_PARAMS_MALE = ModelParams(1 / 15, 1 / 230, 1, 1.15, -0.28, 3.19, 6.7)
MODEL_PARAMS_FEMALE = ModelParams(1 / 15, 1 / 230, 1, 1.15, -0.28, 3.19, 6.7)

with eui.nav_panel("Hb and ferritin prediction"):

    @reactive.effect
    def set_default_sex_height_weight():
        if input.sex() == "2":
            weight_val = MALE_WEIGHT
            height_val = MALE_HEIGHT
            fer_val = MALE_FER
            hb_val = MALE_HB
            don_freq_val = 5
        else:
            weight_val = FEMALE_WEIGHT
            height_val = FEMALE_HEIGHT
            fer_val = FEMALE_FER
            hb_val = FEMALE_HB
            don_freq_val = 3

        # when changing sex update everything to mean male/female?
        ui.update_numeric("height", value=height_val)
        ui.update_numeric("weight", value=weight_val)
        ui.update_numeric("fer_base", value=fer_val)
        ui.update_numeric("Hb_base", value=hb_val)
        ui.update_numeric("don_freq", value=don_freq_val)

    # ui.panel_title("Hb and ferritin prediction")

    Hb_final_val = reactive.value()
    fer_final_val = reactive.value()
    Hb_final_val_low = reactive.value()
    fer_final_val_low = reactive.value()
    Hb_final_val_high = reactive.value()
    fer_final_val_high = reactive.value()

    with ui.layout_columns():
        ui.input_slider(
            "don_freq",
            "Donation frequency per year",
            min=1,
            max=5,
            value=3,
            step=1,
            width="100%",
        )
        ui.input_numeric(
            "fer_base", "Ferritin baseline (ng/mL)", FEMALE_FER, min=0, max=10000
        )
        ui.input_numeric(
            "Hb_base", "Hb baseline mmol/L", FEMALE_HB, min=0, step=0.1, max=1000
        )
        ui.input_numeric("ndons", "Number of donations", 5, min=2, max=20, step=1)

    @render.plot
    def plot():
        don_freq = input.don_freq()
        dt = int(365 / don_freq)
        t_end = input.ndons() * dt
        t = np.arange(0, t_end + dt, dt).astype(int)
        tv = np.ones_like(t) * 0.5
        tv[0] = 0.005
        Hb_base = input.Hb_base()
        if Hb_base is None:
            Hb_base = 9
        fer_base = input.fer_base()
        if fer_base is None:
            fer_base = 70

        # TODO set donor data and model params
        if input.sex() == "2":  # males
            BW = input.weight()
            V = blood_volume_func(input.height(), BW, "male")
            Hb_thres = 8.4
            model_params = MODEL_PARAMS_MALE
        else:
            BW = input.weight()
            V = blood_volume_func(input.height(), BW, "female")
            Hb_thres = 7.8
            model_params = MODEL_PARAMS_FEMALE

        donor_data = DonorData(t, tv, V, BW, Hb_base, fer_base)

        Hb, fer = model(donor_data, model_params)

        Hb_final_val.set(Hb[-1])
        fer_final_val.set(fer[-1])

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(t, Hb, "ko", label="donation")
        ax2.plot(t, fer, "ko")

        mask_fer_30 = fer < 30
        mask_fer_15 = fer < 15
        mask_Hb_thres = Hb < Hb_thres

        ax1.plot(t[mask_Hb_thres], Hb[mask_Hb_thres], ls="", color="red", marker="o")
        ax2.plot(t[mask_fer_30], fer[mask_fer_30], ls="", color="orange", marker="o")
        ax2.plot(t[mask_fer_15], fer[mask_fer_15], ls="", color="red", marker="o")

        ax1.plot(t[0], Hb[0], "ko", mfc="white", label="intake")
        ax2.plot(t[0], fer[0], "ko", mfc="white")

        ax1.set(
            xlabel="days since first donation",
            ylabel="Hb [mmol/L]",
            ylim=[Hb.min() * 0.9, Hb.max() + 0.5],
        )
        ax2.set(
            xlabel="days since first donation",
            ylabel="ferritin [ng/mL]",
            ylim=[fer.min() * 0.5, 10 ** (np.log10(fer.max()) + 0.11)],
        )

        ax1.axhline(Hb_thres, ls=":", color="r")
        ax2.axhline(30, ls=":", color="orange")
        ax2.axhline(15, ls=":", color="red")

        if input.uncertainty():
            donor_data_low = DonorData(
                t, tv, V, BW, Hb_base - 0.4, 10 ** (np.log10(fer_base) - 0.1)
            )
            donor_data_high = DonorData(
                t, tv, V, BW, Hb_base + 0.4, 10 ** (np.log10(fer_base) + 0.1)
            )
            model_params_low = ModelParams(
                alpha=model_params.alpha / 1.05,
                beta=model_params.beta / 1.05,
                gamma=model_params.gamma * 1.05,
                kappa=model_params.kappa * 1.05,
                iron_a=model_params.iron_a,
                iron_b=model_params.iron_b,
                iron_c=model_params.iron_c,
            )
            model_params_high = ModelParams(
                alpha=model_params.alpha * 1.05,
                beta=model_params.beta * 1.05,
                gamma=model_params.gamma / 1.05,
                kappa=model_params.kappa / 1.05,
                iron_a=model_params.iron_a,
                iron_b=model_params.iron_b,
                iron_c=model_params.iron_c,
            )

            Hb_low, fer_low = model(donor_data_low, model_params_low)
            ax1.plot(t, Hb_low, "_", color="grey")
            ax2.plot(t, fer_low, "_", color="grey")

            Hb_high, fer_high = model(donor_data_high, model_params_high)
            ax1.plot(t, Hb_high, "_", color="grey")
            ax2.plot(t, fer_high, "_", color="grey")
            Hb_final_val_low.set(Hb_low[-1])
            Hb_final_val_high.set(Hb_high[-1])
            fer_final_val_low.set(fer_low[-1])
            fer_final_val_high.set(fer_high[-1])

        if input.interp():
            t_interp = np.arange(0, t_end, 2)
            t_interp = np.sort(np.unique(np.concatenate([t_interp, t])))
            t
            # t_interp[-1] = t.max()
            tv_interp = np.zeros_like(t_interp, dtype="float")
            tv_interp[np.isin(t_interp, t)] = 0.5
            tv_interp[0] = 0.005

            donor_data_interp = DonorData(t_interp, tv_interp, V, BW, Hb_base, fer_base)
            Hb, fer = model(donor_data_interp, model_params)

            ax1.plot(t_interp, Hb, "k-", alpha=0.4, zorder=0)
            ax2.plot(t_interp, fer, "k-", alpha=0.4, zorder=0)
            if input.uncertainty():

                donor_data_low = DonorData(
                    t_interp,
                    tv_interp,
                    V,
                    BW,
                    Hb_base - 0.4,
                    10 ** (np.log10(fer_base) - 0.1),
                )
                donor_data_high = DonorData(
                    t_interp,
                    tv_interp,
                    V,
                    BW,
                    Hb_base + 0.4,
                    10 ** (np.log10(fer_base) + 0.1),
                )
                Hb_low, fer_low = model(donor_data_low, model_params_low)
                Hb_high, fer_high = model(donor_data_high, model_params_high)
                ax1.fill_between(t_interp, Hb_low, Hb_high, color="grey", alpha=0.3)
                ax2.fill_between(t_interp, fer_low, fer_high, color="grey", alpha=0.3)

    with ui.layout_columns():
        with eui.value_box():
            """"""

            @render.ui
            def Hb_final():
                if input.uncertainty():
                    return (
                        f"Hb after {input.ndons()} donations {Hb_final_val.get():.1f} "
                        f"(68% CI {Hb_final_val_low.get():.1f}-{Hb_final_val_high.get():.1f}) mmol/L"
                    )
                else:
                    return f"Hb after {input.ndons()} donations {Hb_final_val.get():.1f} mmol/L"

        with eui.value_box():
            """"""

            @render.ui
            def fer_final():
                if input.uncertainty():
                    return (
                        f"Ferritin after {input.ndons()} donations {fer_final_val.get():.0f} "
                        f"(68% CI {fer_final_val_low.get():.0f}-{fer_final_val_high.get():.0f}) ng/mL"
                    )
                else:
                    return f"Ferritin after {input.ndons()} donations {fer_final_val.get():.0f} ng/mL"

    with ui.layout_columns():
        ui.input_numeric(
            "weight", "Donor weight [kg]", value=FEMALE_WEIGHT, min=0, max=300
        )
        ui.input_numeric(
            "height", "Donor height [m]", value=FEMALE_HEIGHT, min=0, max=3, step=0.01
        )
        ui.input_radio_buttons("sex", "Sex", {"1": "Female", "2": "Male"})

        ui.input_checkbox("interp", "Interpolate in-between", False)
        ui.input_checkbox("uncertainty", "Uncertainty estimate", False)


def create_long_term_ferritin_table(Hb_base=9, BW=80, V=5, ndons=5, model_params="M"):
    if model_params == "M":
        model_params = MODEL_PARAMS_MALE
    elif model_params == "F":
        model_params = MODEL_PARAMS_FEMALE

    fer_bases = [
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        125,
        150,
        175,
        200,
        300,
        400,
        500,
    ]
    don_freqs = [1, 2, 3, 4, 5]
    output = np.zeros((len(fer_bases), len(don_freqs)))
    for i, don_freq in enumerate(don_freqs):
        dt = int(365 / don_freq)
        t_end = ndons * dt
        t = np.arange(0, t_end, dt).astype(int)
        tv = np.ones_like(t) * 0.5
        for j, fer_base in enumerate(fer_bases):
            donor_data = DonorData(t, tv, V, BW, Hb_base, fer_base)
            Hb, fer = model(donor_data, model_params)
            output[j, i] = round(fer[-1])
    df = pd.DataFrame({"Start ferritin": fer_bases})
    for i, don_freq in enumerate(don_freqs):
        df[f"{don_freq}/year"] = output[:, i]

    return df


with eui.nav_panel("Donation frequency table"):
    table = create_long_term_ferritin_table()

    values = table.values
    cell_styles = [
        {
            "location": "body",
            "cols": [0],  # Specific column
            "style": {"font-weight": "bold"},
        }
    ]

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if j > 0:
                if values[i, j] < 15:
                    cell_styles.append(
                        {
                            "location": "body",
                            "rows": [i],  # Specific row
                            "cols": [j],  # Specific column
                            "style": {"background-color": "red"},
                        }
                    )
                elif values[i, j] < 30:
                    cell_styles.append(
                        {
                            "location": "body",
                            "rows": [i],  # Specific row
                            "cols": [j],  # Specific column
                            "style": {"background-color": "orange"},
                        }
                    )
                else:
                    cell_styles.append(
                        {
                            "location": "body",
                            "rows": [i],  # Specific row
                            "cols": [j],  # Specific column
                            "style": {"background-color": "green"},
                        }
                    )


    ui.markdown(
        "Click **Start ferritin** in the table below to get optimal donation frequency."
    )

    @render.ui
    def rows():
        rows = table_df.cell_selection().get("rows", [])
        row = table.iloc[rows]
        start_ferritin = row["Start ferritin"]
        if not isinstance(start_ferritin, (float, int)):
            return ""
        vals = row[1:].values
        dfs = table.columns[1:]
        optimal_df = dfs.max()
        end_fer = None
        for i, val in enumerate(vals):
            if val < input.fer_cutoff():
                if i == 0:
                    optimal_df = "0/year"
                else:
                    optimal_df = dfs[i - 1]
                end_fer = vals[i - 1]
                break
        if i == 0:
            end_fer = start_ferritin
        if end_fer is None:
            end_fer = vals[-1]
        return ui.markdown(
            f"""
            Optimal donation frequency **{optimal_df}** for start ferritin: {start_ferritin} ng/mL.
            <br>
            Long-term ferritin **{end_fer} ng/mL** for this donation frequency
            """
        )

    @render.data_frame
    def table_df():
        return render.DataGrid(
            table,
            styles=cell_styles,
            selection_mode="row",
            height="100%",
            # width="500px",
        )

    ui.input_numeric(
        "fer_cutoff",
        "Minimum long-term ferritin level (to stay above)",
        value=30,
        min=0,
        max=1000,
    )
