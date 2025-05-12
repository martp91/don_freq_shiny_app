from collections import namedtuple
from copy import copy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

from shiny import render, reactive
from shiny.express import input
from shiny.express import ui  # as ui
from shiny import ui as uis

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

# TODO: Set male/female best params
MODEL_PARAMS_MALE = ModelParams(1 / 13, 1 / 265, 0.78, 1.27, -0.28, 3.19, 6.7)
MODEL_PARAMS_FEMALE = ModelParams(1 / 18, 1 / 196, 1.21, 1.07, -0.28, 3.19, 6.7)

# TODO: ui.tooltip hover info boxes


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
            Hb_base=donor_data_high.Hb_base + 0.4,
            fer_base=donor_data_high.fer_base * 1.1,
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
            donor_data_low = donor_data_low._replace(
                don_times=t_interp, taken_vol=tv_interp
            )
            donor_data_high = donor_data_high._replace(
                don_times=t_interp, taken_vol=tv_interp
            )
            Hb_low_interp, fer_low_interp = model(donor_data_low, model_params_low)
            Hb_high_interp, fer_high_interp = model(donor_data_high, model_params_high)

    if ax1 is not None:

        ax1.plot(t, Hb, "ko", label="donation")
        mask_Hb_thres = Hb < float(Hb_thres.get())
        ax1.plot(t[mask_Hb_thres], Hb[mask_Hb_thres], ls="", color="red", marker="o")
        ax1.plot(t[0], Hb[0], "ko", mfc="white", label="intake")
        ax1.set(
            xlabel="days since first donation",
            ylabel="Hb [mmol/L]",
            ylim=[Hb.min() * 0.9, Hb.max() + 0.5],
        )
        ax1.axhline(float(Hb_thres.get()), ls=":", color="r")
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
                
    if ax1 is None:
        return ax2
    if ax2 is None:
        return ax1
    return ax1, ax2

 


with ui.nav_panel("Hb and ferritin prediction"):


    Hb_final_val = reactive.value()
    fer_final_val = reactive.value()
    Hb_final_val_low = reactive.value()
    fer_final_val_low = reactive.value()
    Hb_final_val_high = reactive.value()
    fer_final_val_high = reactive.value()
    
    Hb_thres = reactive.value()

    @reactive.calc
    def calc_donor_model():
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
        # print(input.viewport_width())
        # TODO set donor data and model params
        BW = input.weight()
        V = blood_volume_func(input.height(), BW, input.sex())
        if input.sex() == "Male":  # males
            Hb_thres.set(8.4)
            model_params = MODEL_PARAMS_MALE
        else:
            Hb_thres.set(7.8)
            model_params = MODEL_PARAMS_FEMALE

        donor_data = DonorData(t, tv, V, BW, Hb_base, fer_base)
        return donor_data, model_params
        
        
    with ui.card(full_screen=True, fill=False):
        with ui.layout_columns():
            ui.input_checkbox("show_ferritin", "Ferritin", True)
            ui.input_checkbox("show_Hb", "Hb", False)
        with ui.layout_columns():
            with ui.panel_conditional("input.show_ferritin"):
                #TODO: make sure these are not recalculated?
                @render.plot
                def plot_fer():  # Match the unique ID here
                    donor_data, model_params = calc_donor_model()
                    f, ax = plt.subplots(1)
                    plot_Hb_ferritin((None, ax), donor_data, model_params, Hb_thres)

            with ui.panel_conditional("input.show_Hb"):
                @render.plot
                def plot_Hb():  # Match the unique ID here
                    donor_data, model_params = calc_donor_model()
                    if input.show_Hb():
                        f, ax = plt.subplots(1)
                        plot_Hb_ferritin((ax, None), donor_data, model_params, Hb_thres)
                

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
        color = "green"
        if fer_final_val.get() < 15:
            color = "red"
        elif fer_final_val.get() < 30:
            color = "orange"
        fer_text = f"Ferritin after {input.ndons()} donations <span style='color:{color}'>**{fer_final_val.get():.0f} ng/mL**</span>"
        Hb_text = (
            f"Hb after {input.ndons()} donations **{Hb_final_val.get():.1f} mmol/L**"
        )
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
        ui.input_switch("interp", "Show in-between donations", False)

    @render.text
    def _():
        return f"Showing simulated {input.sex()} donor with weight of {input.weight()} kg and height {input.height()} m"
    
    ui.input_action_button("toggle_button", "Show/Hide Extra inputs")


    with ui.panel_conditional(
        "input.toggle_button % 2 == 1",  # Show when button is clicked an odd number of times
    ):
        with ui.layout_columns(): 
            ui.input_numeric(
                "ndons", "Number of donations", 5, min=2, max=20, step=1
            )
            ui.input_switch("uncertainty", "Uncertainty estimate", False),
            ui.input_radio_buttons(
                "sex", "Sex", {"Female": "Female", "Male": "Male"}
            )
            ui.input_numeric(
                "weight", "Donor weight [kg]", value=FEMALE_WEIGHT, min=0, max=300
            )
            ui.input_numeric(
                "height",
                "Donor height [m]",
                value=FEMALE_HEIGHT,
                min=1,
                max=3,
                step=0.01,
            )
    

def create_long_term_ferritin_table(years_future=2, sex="Male"):
    # TODO: set sex to change these values
    if sex == "Male":
        Hb_base = MALE_HB
        BW = MALE_WEIGHT
        V = blood_volume_func(MALE_HEIGHT, BW, "male")
        model_params = MODEL_PARAMS_MALE
    elif sex == "Female":
        Hb_base = FEMALE_HB
        BW = FEMALE_WEIGHT
        V = blood_volume_func(FEMALE_HEIGHT, BW, "female")
        model_params = MODEL_PARAMS_FEMALE
    else:
        raise ValueError(f"sex should be any of Male/Female was {sex}")

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


with ui.nav_panel("Donation frequency table"):

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

        color = "green"
        if end_fer < 15:
            color = "red"
        elif end_fer < 30:
            color = "orange"
        return ui.markdown(
            f"""
            Optimal donation frequency **{optimal_df}** for start ferritin: {start_ferritin} ng/mL.
            <br>
            Long-term ferritin <span style='color:{color}'>**{end_fer} ng/mL**</span> for this donation frequency
            """
        )

    @render.data_frame
    def table_df():
        table = create_long_term_ferritin_table(
            years_future=input.years_future(), sex=input.sex_df()
        )

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
        value=20,
        min=0,
        max=1000,
    )
    ui.input_action_button("toggle_button_df", "Show/Hide Extra inputs"),
    uis.panel_conditional(
        "input.toggle_button_df % 2 == 1",  # Show when button is clicked an odd number of times
        uis.layout_columns(
            [
                ui.input_numeric(
                    "years_future", "Years into the future", 2, min=1, max=20, step=1
                ),
                ui.input_radio_buttons(
                    "sex_df", "Sex", {"Male": "Male", "Female": "Female"}
                ),
            ]
        ),
    )
