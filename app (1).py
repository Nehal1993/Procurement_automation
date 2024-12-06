
import gradio as gr
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress specific warnings related to ARIMA
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Function to process the CSV and perform the forecasting
def demand_forecasting(csv_file, type_value):
    # Load the CSV file
    df = pd.read_csv(csv_file.name)  # .name is used to get the file path

    # Preprocess the data
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    df['Date of Sale'] = pd.to_datetime(df['Date of Sale'], format='%d-%b-%y')

    # Filter the data based on 'Type'
    df_filtered = df[df['Type'] == type_value]
    df_filtered.set_index('Date of Sale', inplace=True)

    # Group and resample data by product name and month-end frequency
    monthly_sales_by_product = (
        df_filtered.groupby('Product Name')
                   .resample('M')['Quantity']
                   .sum()
                   .reset_index()
    )

    # Calculate total sales by product
    Total_sales_product = df_filtered.groupby('Product Name')['Quantity'].sum().reset_index()

    # Create a list to store the forecasted sums for each product
    forecast_sums = []

    # Get a list of unique product names
    product_names = monthly_sales_by_product['Product Name'].unique()

    # Loop through each product and forecast sales
    for product_name in product_names:
        # Filter data for the current product
        product_data = monthly_sales_by_product[monthly_sales_by_product['Product Name'] == product_name].copy()

        # Ensure 'Date of Sale' is in datetime format using .loc[]
        product_data.loc[:, 'Date of Sale'] = pd.to_datetime(product_data['Date of Sale'])

        # Set the 'Date of Sale' as the index and ensure the index has frequency information
        product_data.set_index('Date of Sale', inplace=True)

        # Check if we have enough data points and set a frequency
        if len(product_data) >= 1:  # Adjust based on ARIMA requirements
            product_data = product_data.asfreq('M')  # Set frequency to monthly

            try:
                # Fit the ARIMA model
                model = ARIMA(product_data['Quantity'], order=(1, 1, 1))
                model_fit = model.fit()

                # Forecast for the next 4 months
                forecast_steps = 4
                forecast = model_fit.forecast(steps=forecast_steps)


                # Sum the forecasted values for this product
                forecast_sum = forecast.sum()
                forecast_sum = round(forecast_sum)
                forecast_sums.append({'Product Name': product_name, 'Forecasted Quantity Sum': forecast_sum})

            except Exception as e:
                print(f"Could not fit ARIMA model for {product_name}: {e}")
        else:
            print(f"Not enough data points for {product_name} to fit ARIMA model.")

    # Create a DataFrame from the forecast sums
    forecast_summary_df = pd.DataFrame(forecast_sums)

    # Perform a left merge to ensure all product names from Total_sales_product are included
    combined_df = pd.merge(Total_sales_product, forecast_summary_df, on='Product Name', how='left')

    # Rename columns for clarity
    combined_df.columns = ['Product Name', 'Total Quantity Sold', 'Forecasted Quantity Sum']

    # Save the combined DataFrame to a CSV file
    output_file = 'combined_df.csv'
    combined_df.to_csv(output_file, index=False)

    return output_file

# Gradio Interface
interface = gr.Interface(
    fn=demand_forecasting,
    inputs=[
        gr.File(label="Upload CSV File"),
        gr.Textbox(label="Enter 'Type' (e.g., 'EW')")
    ],
    outputs=gr.File(label="Download Combined Data CSV")
)

# Launch the interface
interface.launch(share=True)
