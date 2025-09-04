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


# https://icons.getbootstrap.com/icons/question-circle-fill/
question_circle_fill = uis.HTML(
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle-fill mb-1" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.496 6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247zm2.325 6.443c.61 0 1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/></svg>'
)


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


# MODEL_PARAMS_MALE = ModelParams(1 / 13, 1 / 265, 0.78, 1.27, -0.28, 3.19, 6.7)
MODEL_PARAMS_MALE = ModelParams(1 / 10, 1 / 276, 0.8, 1.86, -0.28, 3.19, 6.7)
# MODEL_PARAMS_FEMALE = ModelParams(1 / 18, 1 / 196, 1.21, 1.07, -0.28, 3.19, 7.0)
MODEL_PARAMS_FEMALE = ModelParams(1 / 22, 1 / 172, 1.36, 1.67, -0.28, 3.19, 7.0)
# MODEL_PARAMS_AVE = ModelParams(1/16, 1/208, 1.05, 1.17, -0.28, 3.19, 6.85)
MODEL_PARAMS_AVE = ModelParams(1/15, 1/218, 1.04, 1.77, -0.28, 3.19, 6.85)

AVE_WEIGHT = 79
AVE_HEIGHT = 1.78


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
    
    @reactive.effect
    def set_default_sex_height_weight():
        if input.sex() == "Male":
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
        
    with ui.card(full_screen=True):
        with ui.card_header():
            with ui.tooltip(placement='right'):
                ui.span("Simulated donor carreer", question_circle_fill)
                "This plot shows the predicted ferritin (and Hb) level prior to each whole-blood donation given the parameters set below"
                
        with ui.layout_columns(fill=False):
            ui.input_checkbox_group("show_ferritin_Hb", None, {"Ferritin": "Ferritin", "Hb": "Hb"}, selected=["Ferritin"], inline=True)
        
        with ui.layout_columns():
            with ui.panel_conditional("input.show_ferritin_Hb.includes('Ferritin')"):
                #TODO: make sure these are not recalculated?
                @render.plot
                def plot_fer():  # Match the unique ID here
                    donor_data, model_params = calc_donor_model()
                    f, ax = plt.subplots(1)
                    plot_Hb_ferritin((None, ax), donor_data, model_params, Hb_thres)

            with ui.panel_conditional("input.show_ferritin_Hb.includes('Hb')"):
                @render.plot
                def plot_Hb():  # Match the unique ID here
                    donor_data, model_params = calc_donor_model()
                    f, ax = plt.subplots(1)
                    plot_Hb_ferritin((ax, None), donor_data, model_params, Hb_thres)
                
        with ui.tooltip():
            ui.input_slider(
                "don_freq",
                "Donation frequency per year",
                min=1,
                max=5,
                value=3,
                step=1,
                width="100%",
            )
            "Slide to set the number of donations per year"

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
        ui.input_radio_buttons(
            "sex", "Sex", {"Female": "Female", "Male": "Male"}
        )
        with ui.tooltip():
            ui.input_switch("interp", "Show in-between donations", False)
            "Show the prediction of ferritin/Hb inbetween donations"

        with ui.tooltip():
            ui.input_numeric(
                "fer_base", "Ferritin baseline (ng/mL)", FEMALE_FER, min=0, max=10000
            )
            "The baseline ferritin is the ferritin of a donor before any donations. This determines the iron storage of a donor and is the most influential when determining a donor carreer"
        with ui.tooltip():
            ui.input_numeric(
                "Hb_base", "Hb baseline mmol/L", FEMALE_HB, min=0, step=0.1, max=1000
            )
            "The baseline Hb (Hemoglobin) is the Hb of a donor before any donations"
        ui.input_numeric(
            "ndons", "Number of donations", 5, min=2, max=20, step=1
        )
        ui.input_action_button("toggle_button", "Show/Hide Extra inputs")
        
    @render.text
    def _():
        return f"Showing simulated {input.sex()} donor with weight of {input.weight()} kg and height {input.height()} m. Donating whole-blood {input.don_freq()} times per year."
    


    with ui.panel_conditional(
        "input.toggle_button % 2 == 1",  # Show when button is clicked an odd number of times
    ):
        with ui.layout_columns(): 
            ui.input_switch("uncertainty", "Uncertainty estimate", False),
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
    

def create_long_term_ferritin_table(years_future=2, sex='ave', Hb_base=None,
                                    BW=None, H=None, alpha=None, beta=None,
                                    gamma=None, kappa=None, iron_A=None, iron_B=None, iron_C=None):
    
    if sex == 'Male': 
        V = blood_volume_func(H, BW, 'male')
    elif sex == 'Female':
        V = blood_volume_func(H, BW, 'female')
    else: 
        V = (blood_volume_func(H, BW, 'female') + blood_volume_func(H, BW, 'male'))/2.
    #take inverse of alpha,beta. The naming is shit sorry 
    model_params = ModelParams(1/alpha, 1/beta, gamma, kappa, iron_A, iron_B, iron_C)
        
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
        250,
        300,
        350,
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
    
    @reactive.effect
    def set_default_sex_height_weight_df():
        if input.sex_df() == "Male":
            weight_val = MALE_WEIGHT
            height_val = MALE_HEIGHT
            hb_val = MALE_HB
            V_val = blood_volume_func(height_val, weight_val, 'male')
            alpha_val = MODEL_PARAMS_MALE.alpha
            beta_val = MODEL_PARAMS_MALE.beta
            gamma_val = MODEL_PARAMS_MALE.gamma
            kappa_val = MODEL_PARAMS_MALE.kappa
            iron_A_val = MODEL_PARAMS_MALE.iron_a
            iron_B_val = MODEL_PARAMS_MALE.iron_b
            iron_C_val = MODEL_PARAMS_MALE.iron_c
        elif input.sex_df() == 'Female':
            weight_val = FEMALE_WEIGHT
            height_val = FEMALE_HEIGHT
            hb_val = FEMALE_HB
            alpha_val = MODEL_PARAMS_FEMALE.alpha
            beta_val = MODEL_PARAMS_FEMALE.beta
            gamma_val = MODEL_PARAMS_FEMALE.gamma
            kappa_val = MODEL_PARAMS_FEMALE.kappa
            iron_A_val = MODEL_PARAMS_FEMALE.iron_a
            iron_B_val = MODEL_PARAMS_FEMALE.iron_b
            iron_C_val = MODEL_PARAMS_FEMALE.iron_c
            V_val = blood_volume_func(height_val, weight_val, 'female')
        else:
            weight_val = AVE_WEIGHT
            height_val = AVE_HEIGHT
            hb_val = (FEMALE_HB+MALE_HB)/2.
            alpha_val = MODEL_PARAMS_AVE.alpha
            beta_val = MODEL_PARAMS_AVE.beta
            gamma_val = MODEL_PARAMS_AVE.gamma
            kappa_val = MODEL_PARAMS_AVE.kappa
            iron_A_val = MODEL_PARAMS_AVE.iron_a
            iron_B_val = MODEL_PARAMS_AVE.iron_b
            iron_C_val = MODEL_PARAMS_AVE.iron_c
            

        # when changing sex update everything to mean male/female?
        ui.update_slider("H", value=height_val)
        ui.update_slider("BW", value=weight_val)
        ui.update_slider("Hb_base_df", value=hb_val)
        ui.update_slider("alpha", value=1/alpha_val)
        ui.update_slider("beta", value=1/beta_val)
        ui.update_slider("gamma", value=gamma_val)
        ui.update_slider("kappa", value=kappa_val)
        ui.update_slider("iron_A", value=iron_A_val)
        ui.update_slider("iron_B", value=iron_B_val)
        ui.update_slider("iron_C", value=iron_C_val)

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
            if val <= input.fer_cutoff():
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
            years_future=input.years_future(),
            Hb_base=input.Hb_base_df(), BW=input.BW(), H=input.H(),
            alpha=input.alpha(), beta=input.beta(), gamma=input.gamma(),
            kappa=input.kappa(),
            iron_A=input.iron_A(),  
            iron_B=input.iron_B(),  
            iron_C=input.iron_C(),  
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
    with ui.panel_conditional("input.toggle_button_df % 2 == 1"):
        with ui.layout_columns(): 
            ui.input_numeric("years_future", "Years into the future", 2, min=1, max=20, step=1)
            ui.input_radio_buttons(
                "sex_df", "Sex", {"Male": "Male", "Female": "Female", 'Average': 'Average'}
            )
            ui.input_slider('BW', 'Weight', value=None, min=50, max=150, step=1)
            ui.input_slider('H', 'Height', value=None, min=1.5, max=2.5, step=0.01)
            ui.input_slider('Hb_base_df', 'Hb_base', value=None, min=6, max=13, step=0.1)
            ui.input_slider('alpha', '1/alpha', value=None, min=1, max=100, step=1)
            ui.input_slider('beta', '1/beta', value=None, min=50, max=1000, step=1)
            ui.input_slider('gamma', 'gamma', value=None, min=0.5, max=2, step=0.1)
            ui.input_slider('kappa', 'kappa', value=None, min=0.5, max=3, step=0.1)
            ui.input_slider('iron_A', 'iron_A', value=None, min=-0.4, max=-0.01, step=0.01)
            ui.input_slider('iron_B', 'iron_B', value=None, min=2, max=4, step=0.1)
            ui.input_slider('iron_C', 'iron_C', value=None, min=5, max=9, step=0.1)                
