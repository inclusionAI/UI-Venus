"""Shared typography and Morandi colors for offline README figures."""

INK = "#38434A"
MUTED = "#667179"
RULE = "#DCE1DF"
PAPER = "#F7F5F0"
BLUE = "#B8C7D0"
SAGE = "#C1CBBE"
LAVENDER = "#C7C0D2"
SAND = "#D8CEBB"
TEAL = "#B5C7C4"
TAUPE = "#CEBFC0"
CLAY = "#B87460"
CLAY_DARK = "#774C3F"
SAGE_DARK = "#587565"
BLUE_DARK = "#627D8C"
SAGE_BG = "#EDF1EA"
REFERENCE_COLORS = [BLUE, SAGE, LAVENDER, SAND, TEAL, TAUPE]
SANS = ["Arial", "DejaVu Sans"]
SERIF = ["Georgia", "DejaVu Serif"]


def apply_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": SANS, "font.size": 14,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "svg.fonttype": "path"})
