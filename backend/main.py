import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sklearn.ensemble import RandomForestRegressor # Using Regressor for example, change if classification needed
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware

app = FastAPI()

# Add CORS middleware to allow requests from your frontend
origins = [
    "http://localhost:3000",  # Allow requests from your Next.js frontend
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Variable to store the trained model
model = None
MODEL_FILE = "random_forest_model.pkl"

# Load model if it exists
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.post("/learn")
async def learn_from_csv(file: UploadFile = File(...)):
    """
    Accepts a CSV file, trains a RandomForest model, and stores it.
    Assumes the last column of the CSV is the target variable.
    Attempts to handle non-numeric data by coercing and dropping problematic columns/rows.
    """
    global model

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        # Read the CSV file
        # Use keep_default_na=True and na_values to handle common missing value representations
        df = pd.read_csv(file.file, keep_default_na=True, na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'])


        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty.")

        if df.shape[1] < 2:
             raise HTTPException(status_code=400, detail="CSV must contain at least one feature column and one target column.")


        # Assume the last column is the target variable
        # Select all columns except the last for features
        X = df.iloc[:, :-1].copy() # Use .copy() to avoid SettingWithCopyWarning
        # Select the last column for the target
        y = df.iloc[:, -1].copy() # Use .copy()

        # --- Data Cleaning and Type Conversion ---

        # List to keep track of columns to drop
        cols_to_drop = []

        # Process feature columns (X)
        for col in X.columns:
            # Attempt to convert column to numeric, coercing errors to NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')

            # Check if the column still contains non-numeric strings after coercion
            # If the original dtype was 'object' (likely strings) and it still has NaNs after coercing,
            # it means there were non-numeric strings that couldn't be converted.
            # We'll drop such columns for simplicity in this example.
            if X[col].isnull().any() and df[col].dtype == 'object':
                 print(f"Warning: Column '{col}' contains non-numeric strings that could not be converted. Dropping column.")
                 cols_to_drop.append(col)

        # Drop identified columns from X
        X = X.drop(columns=cols_to_drop)

        # Process target column (y)
        # Attempt to convert target to numeric, coercing errors to NaN
        y = pd.to_numeric(y, errors='coerce')

        # Drop rows where the target variable is NaN (missing or couldn't be converted)
        original_rows = df.shape[0]
        rows_before_drop = X.shape[0]
        nan_target_indices = y[y.isnull()].index
        X = X.drop(index=nan_target_indices)
        y = y.drop(index=nan_target_indices)
        rows_after_drop = X.shape[0]

        if rows_before_drop > rows_after_drop:
            print(f"Warning: Dropped {rows_before_drop - rows_after_drop} rows due to non-numeric or missing target values.")


        # --- End Data Cleaning ---


        if X.empty or y.empty:
             raise HTTPException(status_code=400, detail="After cleaning, no valid numerical data remains for training. Please check your CSV.")

        # Handle remaining missing values in features (if any) - Simple imputation (fill with mean)
        # You might need a more sophisticated strategy depending on your data
        if X.isnull().any().any():
            print("Warning: Missing values found in features after dropping non-numeric columns. Filling with column mean.")
            X = X.fillna(X.mean())


        # Train the RandomForest model
        # You might need to adjust parameters based on your data
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Save the trained model
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)

        return JSONResponse(content={"message": "Model trained and saved successfully!"})

    except Exception as e:
        # Log the error for debugging
        print(f"Error during training: {e}")
        # Provide a more user-friendly error message if possible,
        # but include the original error detail for debugging.
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {e}")


@app.get("/ask")
async def ask_with_model(q: str):
    """
    Accepts a question (as a query parameter 'q'), uses the trained model
    to generate a prediction, and returns the result.
    Assumes the query 'q' can be converted into a format the model understands.
    This is a simplified example; real-world scenarios need more complex query processing.
    For this example, we'll assume 'q' is a comma-separated string of feature values.
    """
    if model is None:
        raise HTTPException(status_code=404, detail="Model not trained yet. Please upload a CSV file first.")

    try:
        # Convert the query string 'q' into a format the model expects
        # This is a placeholder - you'll need to adapt this based on your data and model
        # Example: assuming 'q' is "1.2,3.4,5.6" for 3 features
        try:
            # Attempt to convert query string to a list of floats
            features = [float(x.strip()) for x in q.split(',')]
            # Convert list to DataFrame for prediction
            features_array = pd.DataFrame([features])
        except ValueError:
            # If conversion to float fails, it's an invalid format
            raise HTTPException(status_code=400, detail="Invalid query format. Expected comma-separated numbers.")

        # Ensure the number of features matches the model's expectation
        # model.n_features_in_ is available after the model has been trained
        if features_array.shape[1] != model.n_features_in_:
             raise HTTPException(status_code=400, detail=f"Incorrect number of features provided. Expected {model.n_features_in_}, but received {features_array.shape[1]}. Please provide {model.n_features_in_} comma-separated numbers.")


        # Make a prediction
        prediction = model.predict(features_array)

        # Return the prediction. If it's an array, return the first element.
        # Ensure the result is JSON serializable (e.g., convert numpy types)
        result = float(prediction[0]) if isinstance(prediction, (list, pd.Series)) and len(prediction) > 0 else float(prediction)


        return JSONResponse(content={"question": q, "prediction": result})

    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# Basic root endpoint
@app.get("/")
async def read_root():
    return {"message": "FastAPI ML Backend is running."}

# Run the app with: uvicorn main:app --reload