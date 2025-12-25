import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr

def run_analysis():
    # Load dataset
    df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

    # Data cleaning
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[df["Quantity"] > 0]

    # Feature engineering
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Select numeric columns
    numeric_df = df[["Quantity", "UnitPrice", "TotalPrice"]]

    # Correlation matrix
    corr = numeric_df.corr()


    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Correlation Matrix of Purchase Variables")
    plt.tight_layout()

    # Save plot
    plot_path = "correlation.png"
    plt.savefig(plot_path)
    plt.close()

    #sample for pairplot(to avoid heavy computation)
    pairplot_df = numeric_df.sample(1000,random_state=42)

    #create pairplot
    sns.pairplot(pairplot_df)
    pairplot_path = "pairplot.png"
    plt.savefig(pairplot_path)
    plt.close()


    info_text = f"""
Dataset Shape after Cleaning: {df.shape}
Columns Used: Quantity, UnitPrice, TotalPrice
"""

    return info_text, plot_path , pairplot_path


# Gradio UI
app = gr.Interface(
    fn=run_analysis,
    inputs=[],
    outputs=[
        gr.Textbox(label="Dataset Information"),
        gr.Image(label="Correlation Heatmap"),
        gr.Image(label="Pair Plot(sampled)")
    ],
    title="Customer Purchase Behavior Analysis",
    description="EDA and visualization of customer purchase patterns using the Online Retail dataset."
)

app.launch()

