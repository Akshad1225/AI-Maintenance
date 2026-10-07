# AI Maintenance Planner

This version upgrades the original project into a usable predictive-maintenance dashboard.

## What it does

1. Loads the full 10,000-row AI4I 2020 dataset.
2. Trains a Random Forest classifier using air temperature, rotational speed, torque and tool wear.
3. Calculates failure risk for each machine.
4. Calculates an operational criticality proxy.
5. Combines risk + criticality into a maintenance priority score.
6. Produces maintenance windows, technician requirements and spare-parts urgency.
7. Provides a **What-if Simulation** for delayed maintenance.
8. Provides model performance metrics including recall and F1.

## Run the dashboard

From this project folder:

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The dashboard opens in your browser.

## Important project limitation

The AI4I 2020 dataset is a public benchmark rather than Suzlon's live plant history. The What-if feature is therefore a transparent scenario simulation: it adjusts sensor values according to user-selected assumptions and reruns the model. These assumptions should be calibrated using real maintenance history before production use.
