import streamlit as st

PALETTES = {
    "Classic": [
        "#2B5C8A", "#4F81AF", "#77ACD3", "#D9D5C9", "#F69035"
    ],
    "Ocean Breeze": [
        "#03045E", "#0077B6", "#00B4D8", "#90E0EF", "#CAF0F8"
    ],
    "Olive Garden Feast": [
        "#606C38", "#283618", "#FEFAE0", "#DDA15E", "#BC6C25"
    ],
    "Fiery Ocean": [
        "#780000", "#C1121F", "#FDF0D5", "#003049", "#669BBC"
    ],
    "Refreshing Summer Fun": [
        "#8ECAE6", "#219EBC", "#023047", "#FFB703", "#FB8500"
    ],
    "Golden Summer Fields": [
        "#CCD5AE", "#E9EDC9", "#FEFAE0", "#FAEDCD", "#D4A373"
    ],
    "Deep Sea": [
        "#0D1B2A", "#1B263B", "#415A77", "#778DA9", "#E0E1DD"
    ],
    "Bold Berry": [
        "#F9DBBD", "#FFA5AB", "#DA627D", "#A53860", "#450920"
    ],
    "Fresh Greens": [
        "#D8F3DC", "#95D5B2", "#52B788", "#2D6A4F", "#1B4332"
    ],
    "Deep Sea": [
        "#EDEDE9", "#D6CCC2", "#F5EBE0", "#E3D5CA", "#D5BDAF"
    ],
}

def vis_palette(agent):

    choice = st.selectbox("请选择配色方案", list(PALETTES.keys()))
    colors = PALETTES[choice]
    
    cols = st.columns(len(colors))
    for col, code in zip(cols, colors):
        col.markdown(
            f"""
            <div style="
                background-color: {code};
                height: 30px;
                border-radius: 4px;
                margin-bottom: 2px;
            "></div>
            <div style="text-align: center; font-size: 10px;">{code}</div>
            """,
            unsafe_allow_html=True
        )
        
    agent.save_color(colors)
        
    return colors
