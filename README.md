# Student Service Request Monitoring System — Streamlit

Section B deployment project for Systems Analysis and Design.

## Run locally

1. Install Python 3.11 (recommended).
2. Open a terminal in this project folder.
3. Create and activate a virtual environment.
4. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Run:

   ```bash
   streamlit run app.py
   ```

The application opens in your browser. It loads `sample_cleaned_data.csv` as demo data. In the Dataset page, upload an approved CSV to replace it.

## Midterm model continuity

The saved `models/trained_model.joblib` is a fitted preprocessing + Decision Tree pipeline reproduced using the uploaded midterm notebook's exact feature list, `train_test_split(test_size=0.20, random_state=42, stratify=y)`, median numeric imputation, most-frequent categorical imputation, OneHotEncoder(handle_unknown="ignore"), and DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42).

Target: `SLA_Breached`

Predictors:
- Service_Type
- Requester_Type
- Requested_Channel
- Priority
- Assigned_Office
- Queue_Length
- Required_Documents_Complete
- SLA_Days
- Submission_Weekday
- Queue_Category

`Request_ID` is an identifier, not a model feature. `Processing_Days`, `Status`, `Reopened`, `Satisfaction_Score`, and `SLA_Breached` are not inference features. `Status` is only used to select pending requests for the staff review queue. The model was fitted on completed records with a known SLA outcome, matching the midterm notebook.

## Dataset CSV schema

Required columns: `Request_ID`, all ten predictors above, and `SLA_Breached`. The app can derive `Submission_Weekday` from `Submission_Date` and `Queue_Category` from `Queue_Length` if those derived columns are missing, but the final required model schema must be present after preparation.

## Deployment to Streamlit Community Cloud

1. Create a GitHub repository and upload the project files.
2. Do not upload real student records, credentials, or private information. The included sample data must be checked for authorization before public hosting; replace it with synthetic data if necessary.
3. Visit https://share.streamlit.io and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Deploy. If the build fails, inspect the app logs and confirm that `requirements.txt` and all project folders are committed.
6. Test all six navigation pages and record screenshots.

## Model evaluation (reproduced from midterm code)

- Training rows: 242
- Test rows: 61
- Accuracy: 75.41%
- Breach precision: 47.37%
- Breach recall: 64.29%
- Breach F1-score: 54.55%
- Confusion matrix, actual rows / predicted columns, class order 0 then 1: `[[37, 10], [5, 9]]`

These values describe the fixed midterm test split, not guaranteed live performance. Probability scores are not claimed to be calibrated.

## Human review and limitations

Predictions are advisory. Authorized staff must verify the underlying request, documents, queue conditions, priority, and SLA target before deciding follow-up. Never automatically reject a request or deny student service based on a prediction. Use synthetic or properly authorized data for public deployment.

## Project files

- `app.py` — Streamlit navigation and page rendering
- `modules/` — data validation, EDA, inference, and dashboard functions
- `models/trained_model.joblib` — fitted pipeline
- `models/model_metrics.json` — reproduced midterm test metrics
- `assets/use_case_diagram.png` — updated use case diagram
- `sample_cleaned_data.csv` — uploaded midterm workbook's cleaned sheet, for local testing only
- `SectionB_BALSIMO_GLESSY_ExamV2.ipynb` — original midterm notebook
