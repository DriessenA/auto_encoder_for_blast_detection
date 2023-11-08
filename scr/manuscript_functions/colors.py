import matplotlib.cm as cm

# from matplotlib.colors import rgb2hex, hex2color
import plotly.express as px
import numpy as np

# CMAP of 24 colors
cmap_24 = px.colors.qualitative.Dark24

# Color blind friendly default map
default_cmap = ["#89AAFF", "#433488", "#DC2626", "#8E8E8E", "#ECDF07"]
default_hex = {
    "lightblue": "#89AAFF",
    "blue": "#433488",
    "red": "#DC2626",
    "grey": "#8E8E8E",
    "yellow": "#ECDF07",
}
default_rgba = {
    "lightblue": (0.5372549019607843, 0.6666666666666666, 1.0, 1.0),
    "blue": (0.2627450980392157, 0.20392156862745098, 0.5333333333333333, 1.0),
    "red": (0.8627450980392157, 0.14901960784313725, 0.14901960784313725, 1.0),
    "grey": (0.5568627450980392, 0.5568627450980392, 0.5568627450980392, 1.0),
    "yellow": (0.9254901960784314, 0.8745098039215686, 0.027450980392156862, 1.0),
}
seq_rgba = cm.plasma(np.linspace(0, 1, 4))

# Cell type maps
healthy_ct_cmap = {
    "HSPCs": seq_rgba[3],
    "Monoblasts/Myeloblasts": seq_rgba[2],
    "Pro-Monocyte": seq_rgba[1],
    "Monocyte": seq_rgba[0],
    "debris": default_rgba["grey"],
}

nh_ct_cmap = {
    "Monocyte": healthy_ct_cmap["Monocyte"],
    "Pro-Monocyte?": healthy_ct_cmap["Pro-Monocyte"],
    "Not_defined": healthy_ct_cmap["debris"],
    "Blast": default_rgba["yellow"],
}

nonmyeloid_cmap = {
    "Basophil": default_hex["blue"],
    "B_cell": default_hex["red"],
    "Erythroblast": default_hex["yellow"],
    "NK_cell": default_hex["grey"],
    "pDC": default_hex["lightblue"],
    "T_cell": "#000000",  # Black
}

# Highlighting maps
blast_vs_rest = {
    "Rest": default_hex["grey"],
    "Blast": default_hex["red"],
}

# Timepoints and predictions

diagnosis_col = default_hex["red"]
remission_col = default_hex["lightblue"]
relapse_col = default_hex["blue"]

timepoints_cmap = {
    "Diagnosis": diagnosis_col,
    "EOI I": remission_col,
    "Remission": remission_col,
    "Relapse": relapse_col,
    "Patient relapse": relapse_col,
    "Patient diagnosis": diagnosis_col,
    "All remission": remission_col,
}

remission_and_prediction = {
    "Remission": remission_col,
    "Predicted healthy": default_hex["red"],
    "Predicted blast": default_hex["yellow"],
    "Unannotated": default_hex["grey"],
    "Missed blast": default_hex["red"],
    "Correct blast": default_hex["yellow"],
}

# Patient_map
patients = [
    "PAUMTZ",
    "PAUZTH",
    "PAUZVP",
    "PAVDZK",
    "PAVEDT",
    "PAVRJP",
    "PAVSFB",
    "PAVTDU",
    "PAVTLN",
    "PAVZEC",
    "PAWBSJ",
    "PAWHCT",
    "PAWHML",
    "PAWWEE",
    "PAWWZL",
    "PAWXIA",
    "PAWZZN",
    "PAXJXC",
    "PAXMIJ",
    "PAXMLI",
    "TEST12378",
]


id_to_patient = {
    "P1": "PAUMTZ",
    "P2": "PAVRJP",
    "P3": "PAXMLI",
    "P4": "PAUZTH",
    "P5": "PAVSFB",
    "P6": "PAWHCT",
    "P7": "PAXJXC",
    "P8": "PAWBSJ",
    "P9": "PAWWEE",
    "P10": "PAWWZL",
    "P11": "PAXMIJ",
    "P12": "PAUZVP",
    "P13": "PAVDZK",
    "P14": "PAVEDT",
    "P15": "PAVTDU",
    "P16": "PAVTLN",
    "P17": "PAVZEC",
    "P18": "PAWXIA",
    "P19": "PAWHML",
    "P20": "PAWZZN",
    "P21": "TEST12378",
}

patient_to_id = {v: k for k, v in id_to_patient.items()}

# patient_cmap = dict(zip(patients, cmap_24[: len(patients)]))
id_cmap = {
    "P1": "#DC050C",
    "P2": "#B17BA6",
    "P3": "#999999",
    "P4": "#FB8072",
    "P5": "#FF7F00",
    "P6": "#B2DF8A",
    "P7": "#BEAED4",
    "P8": "#33A02C",
    "P9": "#8DD3C7",
    "P10": "#A6761D",
    "P11": "#666666",
    "P12": "#1965B0",
    "P13": "#7BAFDE",
    "P14": "#882E72",
    "P15": "#FDB462",
    "P16": "#CFE3E2",
    "P17": "#E78AC3",
    "P18": "#E6AB02",
    "P19": "#55A1B1",
    "P20": "#7570B3",
    "P21": "#AA8282",
}
