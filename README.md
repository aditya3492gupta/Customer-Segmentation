# Flask Customer Segmentation Application

This Flask application is designed for customer segmentation using the K-Means clustering algorithm and RFM (Recency, Frequency, Monetary) analysis. It provides a user-friendly interface to upload customer data, perform clustering, and visualize the results.

## Features

- **Customer Segmentation:** Utilizes K-Means clustering to categorize customers into distinct segments based on their purchase behavior.

- **RFM Analysis:** Incorporates Recency, Frequency, and Monetary analysis to understand and quantify customer engagement.

- **Data Preprocessing:** Handles data cleaning, including handling missing values and outliers, to ensure accurate results.

- **Visualization:** Generates visualizations, including strip plots, to represent the clusters' distribution across different metrics.

## Prerequisites

Before running the application, ensure you have the necessary dependencies installed. You can install them using the following:

```bash
pip install Flask pandas numpy scikit-learn seaborn matplotlib
```

## Usage

1. Clone the repository:

   ```
   gh repo clone aditya3492gupta/Customer-Segmentation

   ```

2. Run the Flask application:

   ```
   python app.py
   ```

   The application will be accessible at `http://127.0.0.1:5000/` in your web browser.

3. Upload your customer dataset in CSV format using the provided form.

4. Explore the clustered results and visualizations on the results page.

## File Structure

- `app.py`: Flask application script containing the main logic and routes.
- `templates/`: Folder containing HTML templates for the user interface.
- `static/`: Folder for storing static files such as images and stylesheets.

## Future Enhancements

The project has potential for further enhancements:

- **User Authentication:** Implement user authentication for secure access.
- **Real-Time Analysis:** Integrate live data streams for real-time customer segmentation.
- **Deployment:** Deploy the application to a production environment for scalability.

## Demo
![Demo](static/demo.gif)

Feel free to explore additional features and improvements based on your project goals.

