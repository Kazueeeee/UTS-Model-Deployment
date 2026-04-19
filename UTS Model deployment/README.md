# Student Placement Project

Files:
- `student_placement_eda_modeling.ipynb` — EDA and modeling notebook
- `train_pipeline_mlflow.py` — sklearn pipeline training script with MLflow logging
- `streamlit_app.py` — Streamlit monolithic deployment app
- `feature_engineering.py` — shared feature engineering utilities
- `artifacts/best_placement_model.pkl` — saved placement classifier
- `artifacts/best_salary_model.pkl` — saved salary regressor
- `artifacts/metrics.json` — saved evaluation summary

## Local run

```bash
pip install -r requirements.txt
python train_pipeline_mlflow.py --features A.csv --targets A_targets.csv
streamlit run streamlit_app.py
```

## Notes
- Salary regression is trained only on `Placed` students because `salary_lpa` is 0 for `Not Placed`.
- The Streamlit app expects `A.csv`, `A_targets.csv`, and `artifacts/` to be present in the same project folder.
