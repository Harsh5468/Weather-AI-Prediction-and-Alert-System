# Weather AI: Prediction and Alert System 🌦️

This project is a web-based application that uses machine learning to predict weather conditions, specifically temperature and the likelihood of rain. Based on these predictions, it provides users with relevant alerts and recommendations for potential weather-related disasters.

## ✨ Features

* **Temperature Prediction:** Utilizes a Random Forest Regressor to predict the temperature based on various atmospheric inputs.
* **Rainfall Prediction:** Employs a Random Forest Classifier to determine if it will rain.
* **Disaster Recommendation System:** Issues alerts for extreme heat, freezing conditions, and heavy rainfall.
* **Interactive Web Interface:** A user-friendly interface built with Flask allows users to input data and receive instant predictions.
* **Prediction History:** Stores and displays a history of past predictions.
* **Responsive Design:** The UI is designed to be accessible on different screen sizes, with a collapsible sidebar for mobile viewing.
* **Light & Dark Mode:** Includes a theme toggle for user comfort.

## 🛠️ How It Works

The application is powered by a Python backend and a simple HTML/CSS frontend.

1.  **Data Processing:** The model is trained on the `weather.csv` dataset. The data is preprocessed by separating features and target variables, and then scaled using `StandardScaler`.
2.  **Machine Learning Models:**
    * A `RandomForestRegressor` is trained to predict continuous temperature values.
    * A `RandomForestClassifier` is trained to predict the binary outcome of whether it will rain or not.
3.  **Backend (Flask):** The Flask server handles web requests.
    * The main route (`/`) renders the `index.html` page and processes the user's input via a POST request.
    * It takes the form data, scales it, feeds it to the trained models, and gets the predictions.
    * The predictions and recommendations are then sent back to be displayed on the `index.html` page.
    * The `/history` route displays past predictions from `predictions.csv`.
4.  **Frontend (HTML/CSS):**
    * `index.html`: The main user interface with a form for inputting weather data.
    * `history.html`: A page to display the prediction history in a table.
    * `style.css`: Provides the styling, including the gradient background, responsive design, and theme modes.

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Python and pip installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content:
    ```
    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn
    Flask
    ```
    Then run:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1.  Execute the main Python script:
    ```sh
    python weather.py
    ```
2.  Open your web browser and go to `http://localhost:5000`.

## 📂 File Structure

```
.
├── weather.py         # Main script with ML models and Flask app
├── templates
│   ├── index.html     # Main page for prediction
│   └── history.html   # Page for prediction history
├── static
│   └── style.css      # CSS styles
├── weather.csv        # Dataset for training
└── predictions.csv    # (Generated) Stores prediction history
```

## 🧑‍💻 Creators

This project was created by:

* **Ansh Patel:** [GitHub](https://github.com/AnshKSP) | [LinkedIn](https://linkedin.com/in/anshpatel2511)
* **Dwij Chauhan:** [GitHub](https://github.com/DwijChauhan) | [LinkedIn](https://www.linkedin.com/in/Dwij-Chauhan/)
* **Harsh Dave:** [GitHub](https://github.com/Harsh5468) | [LinkedIn](https://linkedin.com/in/Dave-Harsh2812)
---

### 🚀 Run It

Install the required packages:

```bash
pip install -r requirements.txt
```
