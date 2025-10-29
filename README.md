# Smartphone Recommendation with K-means Clustering and AHP

A simple web application designed to recommend smartphones based on a user's price range and preference. This project uses K-means clustering to initially filter the options and then applies the Analytic Hierarchy Process (AHP) to rank the results.
This project is intended for educational purposes to demonstrate a simple implementation of K-means and AHP in a recommendation system. The results are representations based on a specific dataset and pre-defined weights, not definitive real-world advice.

The dataset used (`smartphone.xlsx`) is a cleaned and processed version of the "Real-World Smartphones Dataset" available on Kaggle, originally provided by Abhijeet Dahatonde.

---

## Key Features

* **Price Range Filtering**: Allows users to select from five different price brackets.
* **User Preference Selection**: Users can choose their priority:
    * **Performance**: Focuses on CPU, Speed, RAM, and ROM.
    * **Multimedia**: Focuses on Camera, Screen, ROM, and Battery.
    * **All Rounder**: Considers all attributes more or less equally.
* **K-means Clustering**: Automatically groups the smartphones based on the selected preference to find the "best" cluster of phones.
* **AHP Ranking**: Applies the Analytic Hierarchy Process (AHP), a decision-making method using pairwise comparisons, to score and rank the phones within the best cluster.
* **Top 10 Results**: Displays the top 10 recommended smartphones based on the final AHP score.
* **Result Visualization**: Includes a bar chart to visually compare the AHP scores of the top 10 models.
* **Export to Excel**: Allows users to download the final ranked list.

---

## Technologies Used

* **Backend**: Python
    * **Web Framework**: Flask
    * **Data Analysis**: Pandas
    * **Numerical Computing**: Numpy
    * **Machine Learning**: Scikit-learn (for KMeans and MinMaxScaler)
* **Frontend**:
    * HTML5
    * CSS (with Bootstrap)
    * JavaScript
* **Visualization**: Chart.js
* **Data Reading**: openpyxl (required by Pandas for `.xlsx` files)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Python** (Recommended: 3.7 or newer)
* **pip** (Python Package Manager, usually comes with Python)
* **Git** (for cloning the repository)

---

## Installation and Setup

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/ahp_clustering_smartphone.git](https://github.com/your-username/ahp_clustering_smartphone.git)
    cd ahp_clustering_smartphone
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    This project does not include a `requirements.txt` file. You can install the necessary packages manually:
    ```bash
    pip install Flask pandas numpy scikit-learn openpyxl
    ```

4.  **Run the application**:
    ```bash
    python app.py
    ```

5.  **Access the application**:
    Open your web browser and navigate to `http://127.0.0.1:5000` (or `http://localhost:5000`).

---

## How to Use

1.  Open the web application in your browser.
2.  Select your desired price range from the dropdown menu.
3.  Check the box for your preference (Performance, Multimedia, or All Rounder).
4.  Click **Submit**.
5.  You will be shown a table of all smartphones in the "best cluster" for your preference.
6.  Click the **Proceed to AHP** button to see the top 10 ranked results.
7.  On the results page, you can view the ranked list, the comparison chart, and download the data as an Excel file.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  **Fork** the repository on GitHub.
2.  **Clone** your forked repository to your local machine.
3.  Create a new **branch** for your feature or bug fix (`git checkout -b feature/your-feature-name`).
4.  Make your changes and **commit** them (`git commit -m 'Add some feature'`).
5.  **Push** your changes to your fork on GitHub (`git push origin feature/your-feature-name`).
6.  Open a **Pull Request** from your fork to the original repository.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
 
