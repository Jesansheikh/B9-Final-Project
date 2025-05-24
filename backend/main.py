import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sklearn.ensemble import RandomForestRegressor # Using Regressor for example, change if classification needed
import pickle
import os

app = FastAPI()

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
    """
    global model

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        # Read the CSV file
        df = pd.read_csv(file.file)

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty.")

        # Assume the last column is the target variable
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        if X.empty or y.empty:
             raise HTTPException(status_code=400, detail="CSV must contain features and a target variable.")

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
            features = [float(x.strip()) for x in q.split(',')]
            # Reshape for prediction if it's a single sample
            features_array = pd.DataFrame([features])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid query format. Expected comma-separated numbers.")

        # Ensure the number of features matches the model's expectation
        if features_array.shape[1] != model.n_features_in_:
             raise HTTPException(status_code=400, detail=f"Incorrect number of features. Expected {model.n_features_in_}, got {features_array.shape[1]}.")


        # Make a prediction
        prediction = model.predict(features_array)

        # Return the prediction. If it's an array, return the first element.
        result = prediction[0] if isinstance(prediction, (list, pd.Series)) and len(prediction) > 0 else prediction

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

