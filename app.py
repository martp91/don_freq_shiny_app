from collections import namedtuple
from copy import copy
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from shiny import render, ui, reactive
from shiny.express import input
from shiny.express import ui as eui

from model import Hb_fer_model_iron, blood_volume_func

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

#TODO: Set male/female best params
MODEL_PARAMS_MALE = ModelParams(1 / 15, 1 / 230, 1, 1.15, -0.28, 3.19, 6.7)
MODEL_PARAMS_FEMALE = ModelParams(1 / 15, 1 / 230, 1, 1.15, -0.28, 3.19, 6.7)

#TODO: ui.tooltip hover info boxes

def plot_Hb_ferritin(axs, donor_data, model_params, Hb_thres):
    ax1, ax2 = axs
    Hb, fer = model(donor_data, model_params)
    Hb_final_val.set(Hb[-1])
    fer_final_val.set(fer[-1])
    t = donor_data.don_times

    if input.uncertainty():
        donor_data_low = copy(donor_data)
        donor_data_low = donor_data_low._replace(
            Hb_base=donor_data_low.Hb_base - 0.4, fer_base=donor_data_low.fer_base * 0.9
        )
        donor_data_high = copy(donor_data)
        donor_data_high = donor_data_high._replace(
            Hb_base=donor_data_high.Hb_base + 0.4, fer_base=donor_data_high.fer_base * 1.1
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

        Hb_high, fer_high = model(donor_data_high, model_params_high)
        Hb_final_val_low.set(Hb_low[-1])
        Hb_final_val_high.set(Hb_high[-1])
        fer_final_val_low.set(fer_low[-1])
        fer_final_val_high.set(fer_high[-1])

    if input.interp():
        t_interp = np.arange(0, t.max(), 2)
        t_interp = np.sort(np.unique(np.concatenate([t_interp, t])))
        tv_interp = np.zeros_like(t_interp, dtype="float")
        tv_interp[np.isin(t_interp, t)] = 0.5
        tv_interp[0] = 0.005

        donor_data_interp = copy(donor_data)
        donor_data_interp = donor_data_interp._replace(
            don_times=t_interp, taken_vol=tv_interp
        )
        Hb_interp, fer_interp = model(donor_data_interp, model_params)

        if input.uncertainty():
            donor_data_low = donor_data_low._replace(don_times=t_interp, taken_vol=tv_interp)
            donor_data_high = donor_data_high._replace(don_times=t_interp, taken_vol=tv_interp)
            Hb_low_interp, fer_low_interp = model(donor_data_low, model_params_low)
            Hb_high_interp, fer_high_interp = model(donor_data_high, model_params_high)

    if ax1 is not None:

        ax1.plot(t, Hb, "ko", label="donation")
        mask_Hb_thres = Hb < Hb_thres
        ax1.plot(t[mask_Hb_thres], Hb[mask_Hb_thres], ls="", color="red", marker="o")
        ax1.plot(t[0], Hb[0], "ko", mfc="white", label="intake")
        ax1.set(
            xlabel="days since first donation",
            ylabel="Hb [mmol/L]",
            ylim=[Hb.min() * 0.9, Hb.max() + 0.5],
        )
        ax1.axhline(Hb_thres, ls=":", color="r")
        if input.uncertainty():
            ax1.plot(t, Hb_low, "_", color="grey")
            ax1.plot(t, Hb_high, "_", color="grey")
        if input.interp():
            ax1.plot(t_interp, Hb_interp, "k-", alpha=0.4, zorder=0)
            if input.uncertainty():
                ax1.fill_between(
                    t_interp, Hb_low_interp, Hb_high_interp, color="grey", alpha=0.3
                )

    if ax2 is not None:
        ax2.plot(t, fer, "ko")

        mask_fer_30 = fer < 30
        mask_fer_15 = fer < 15

        ax2.plot(t[mask_fer_30], fer[mask_fer_30], ls="", color="orange", marker="o")
        ax2.plot(t[mask_fer_15], fer[mask_fer_15], ls="", color="red", marker="o")

        ax2.plot(t[0], fer[0], "ko", mfc="white")

        ax2.set(
            xlabel="days since first donation",
            ylabel="ferritin [ng/mL]",
            ylim=[fer.min() * 0.5, 10 ** (np.log10(fer.max()) + 0.11)],
        )

        ax2.axhline(30, ls=":", color="orange")
        ax2.axhline(15, ls=":", color="red")
        if input.uncertainty():
            ax2.plot(t, fer_low, "_", color="grey")
            ax2.plot(t, fer_high, "_", color="grey")
        if input.interp():
            ax2.plot(t_interp, fer_interp, "k-", alpha=0.4, zorder=0)
            if input.uncertainty():
                ax2.fill_between(
                    t_interp, fer_low_interp, fer_high_interp, color="grey", alpha=0.3
                )


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

    Hb_final_val = reactive.value()
    fer_final_val = reactive.value()
    Hb_final_val_low = reactive.value()
    fer_final_val_low = reactive.value()
    Hb_final_val_high = reactive.value()
    fer_final_val_high = reactive.value()

    with ui.layout_columns():
        ui.input_checkbox("show_ferritin", "Ferritin", True)
        ui.input_checkbox("show_Hb", "Hb", False)


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
        if input.show_ferritin():
            if input.show_Hb():
                f, (ax2, ax1) = plt.subplots(2, 1, sharex=True)
            else:
                f, ax2 = plt.subplots(1)
                ax1 = None
        else:
            if input.show_Hb():
                f, ax1 = plt.subplots(1)
                ax2 = None
            else:
                return
        
        plot_Hb_ferritin((ax1, ax2), donor_data, model_params, Hb_thres)
        
    ui.input_slider(
        "don_freq",
        "Donation frequency per year",
        min=1,
        max=5,
        value=3,
        step=1,
        width="100%",
    )
                    
    @render.ui
    def Hb_fer_final():
        fer_text = f"Ferritin after {input.ndons()} donations **{fer_final_val.get():.0f} ng/mL**"
        Hb_text = f"Hb after {input.ndons()} donations **{Hb_final_val.get():.1f} mmol/L**"
        if input.show_ferritin():
            if input.show_Hb():
                return ui.markdown(fer_text + "<br>" + Hb_text)
            return ui.markdown(fer_text)
        else:
            if input.show_Hb():
                return ui.markdown(Hb_text)
            return 

    with ui.layout_columns():
        ui.input_numeric(
            "fer_base", "Ferritin baseline (ng/mL)", FEMALE_FER, min=0, max=10000
        )
        ui.input_numeric(
            "Hb_base", "Hb baseline mmol/L", FEMALE_HB, min=0, step=0.1, max=1000
        )
        ui.input_checkbox("interp", "Interpolate in-between", False)
        
    ui.input_action_button("toggle_button", "Show/Hide Extra inputs"),
    ui.panel_conditional(
        "input.toggle_button % 2 == 1",  # Show when button is clicked an odd number of times
        ui.layout_columns([
            ui.input_numeric("ndons", "Number of donations", 5, min=2, max=20, step=1),
            ui.input_checkbox("uncertainty", "Uncertainty estimate", False),
            ui.input_radio_buttons("sex", "Sex", {"1": "Female", "2": "Male"}),
            ui.input_numeric(
                "weight", "Donor weight [kg]", value=FEMALE_WEIGHT, min=0, max=300
            ),
            ui.input_numeric(
                "height", "Donor height [m]", value=FEMALE_HEIGHT, min=1, max=3, step=0.01
            )
        ])
    )



def create_long_term_ferritin_table(Hb_base=9, BW=80, V=5, years_future=2, model_params="M"):
    #TODO: set sex to change these values
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
        ndons = years_future * don_freq
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
    @render.ui
    def make_table():

        return ui.markdown(
            "Click **Start ferritin** in the table below to get optimal donation frequency."
        )

    @render.ui
    def rows():
        rows = table_df.cell_selection().get("rows", [])
        row = table_df.data().iloc[rows]
        start_ferritin = row["Start ferritin"]
        if not isinstance(start_ferritin, (float, int)):
            return ""
        vals = row[1:].values
        dfs = table_df.data().columns[1:]
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
        table = create_long_term_ferritin_table(years_future=input.years_future())

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
    ui.input_action_button("toggle_button_df", "Show/Hide Extra inputs"),
    ui.panel_conditional(
        "input.toggle_button_df % 2 == 1",  # Show when button is clicked an odd number of times
        ui.layout_columns([
            ui.input_numeric("years_future", "Years into the future", 2, min=1, max=20, step=1),
            ui.input_radio_buttons("sex_df", "Sex", {"1": "Female", "2": "Male"}),
        ])
    )
