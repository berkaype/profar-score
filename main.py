import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import six

# Define the data
""" data = {
    "Tesis Adı": ["Ambarlı", "Büyükçekmece", "Selimpaşa", "Silivri", "Çanta"],
    "Çamur Yaşı": [19.95, 10.09, 14.2, 16.18, 13.47],
    "α": [0.5426, 0.8358, 0.6413, 0.7998, 0.9],
    "SAE": [6.46, 4.81, 5.58, 4.44, 3.96],
    "Düzeltilmiş AKM %": [1.2, 13.46, 4.94, 3.61, 52.84]
}"""

data = pd.read_excel("./tesis_data.xlsx")

# Create DataFrame
df = pd.DataFrame(data)

# Interpolate scores
for col in ["Çamur Yaşı", "α", "SAE", "Düzeltilmiş AKM %"]:
    if col == "Çamur Yaşı" or col == "Düzeltilmiş AKM %":
        # For Çamur Yaşı and Düzeltilmiş AKM %, lower values get higher scores
        df[col + " Score"] = np.interp(df[col], (df[col].min(), df[col].max()), (10, 5))
    else:
        # For α and SAE, higher values get higher scores
        df[col + " Score"] = np.interp(df[col], (df[col].min(), df[col].max()), (5, 10))

# Calculate total score and sort
df["Total Score"] = df[["Çamur Yaşı Score", "α Score", "SAE Score", "Düzeltilmiş AKM % Score"]].sum(axis=1)
df_sorted = df.sort_values("Total Score", ascending=False)

df_result = df_sorted[["Tesis Adı", "Total Score"]]


def render_mpl_table_with_precision(data, col_width=3.0, row_height=0.625, font_size=14,
                                    header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                                    bbox=[0, 0, 1, 1], header_columns=0,
                                    ax=None, **kwargs):
    # Format data for precision
    data_formatted = data.copy()
    for col in data.columns[1:]:  # Skip the first column for station names
        data_formatted[col] = data_formatted[col].map('{:,.2f}'.format)
    
    if ax is None:
        size = (np.array(data_formatted.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height * 1.2])  # Adjust size for font
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data_formatted.values, bbox=bbox, colLabels=data_formatted.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax

# Generate the table in a PDF with adjusted precision
# pdf_path_precision = './result.pdf'
# pdf_pages = PdfPages(pdf_path_precision)
# ax = render_mpl_table_with_precision(df_sorted[["Tesis Adı", "Çamur Yaşı Score", "α Score", "SAE Score", "Düzeltilmiş AKM % Score", "Total Score"]], header_columns=0, col_width=2.5)
# pdf_pages.savefig(ax.figure, bbox_inches='tight')
# pdf_pages.close()

def display_table_adjusted_col_width(data):
    data_formatted = data.copy()
    for col in data.columns[1:]:  # Adjust for precision in non-name columns
        data_formatted[col] = data_formatted[col].map('{:,.2f}'.format)
    
    # Calculate column width based on max length of content in each column
    col_widths = [max(data_formatted[col].astype(str).map(len).max(), len(col)) * 0.1 for col in data_formatted.columns]

    row_height = 0.625
    font_size = 14
    header_color = '#40466e'
    row_colors = ['#f1f1f2', 'w']
    edge_color = 'w'
    bbox = [0, 0, 1, 1]
    header_columns = 0

    total_width = sum(col_widths) + len(data_formatted.columns) * 1.2  # Additional space for padding between columns
    size = (total_width, len(data_formatted.index) * row_height + row_height)  # Adjust fig size
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')

    mpl_table = ax.table(cellText=data_formatted.values, bbox=bbox, colLabels=data_formatted.columns, loc='center', colWidths=col_widths)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    plt.show()

# Display the table with adjusted precision using matplotlib
display_table_adjusted_col_width(df_sorted[["Tesis Adı", "Çamur Yaşı Score", "α Score", "SAE Score", "Düzeltilmiş AKM % Score", "Total Score"]])
