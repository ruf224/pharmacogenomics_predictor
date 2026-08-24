import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Train the Pharmacogenomics Dosing Model
@st.cache_data
def train_pgx_model():
    df = pd.read_csv("pharmacogenomics_data.csv")
    
    # Map text values to numbers
    df['Taking_CYP_Inhibitor'] = df['Taking_CYP_Inhibitor'].map({'Yes': 1, 'No': 0})
    
    pheno_map = {'Poor_Metabolizer': 0, 'Intermediate_Metabolizer': 1, 'Normal_Metabolizer': 2, 'UltraRapid_Metabolizer': 3}
    df['CYP2D6_Phenotype'] = df['CYP2D6_Phenotype'].map(pheno_map)
    
    # Target options: Reduce_Dose=0, Maintain_Dose=1, Increase_Dose=2
    target_map = {'Reduce_Dose': 0, 'Maintain_Dose': 1, 'Increase_Dose': 2}
    df['Dose_Adjustment_Required'] = df['Dose_Adjustment_Required'].map(target_map)
    
    features = ['Age', 'Weight_kg', 'CYP2D6_Phenotype', 'Taking_CYP_Inhibitor', 'Base_Dose_mg']
    X = df[features]
    y = df['Dose_Adjustment_Required']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, pheno_map

model, pheno_map = train_pgx_model()

# 2. Design the Multi-User Dashboard Layout
st.title("🧬 Precision Medicine: Personalized Dosing Engine")
st.write("This clinical tool uses AI to cross-reference patient physical dimensions with *CYP2D6* enzyme genotypes to calculate precise, safe medication adjustments.")

st.subheader("📋 Patient Physical Dimensions")
age = st.number_input("Patient Age", min_value=18, max_value=110, value=65)
weight = st.number_input("Patient Weight (kg)", min_value=35, max_value=150, value=70)
base_dose = st.number_input("Standard Intended Drug Base Dose (mg)", min_value=10, max_value=500, value=100)

st.subheader("🧬 Metabolic & Genomic Profile")
genotype = st.selectbox("Patient Evaluated CYP2D6 Genotype Phenotype", list(pheno_map.keys()))
inhibitor = st.selectbox("Is the patient taking a concomitant strong CYP inhibitor?", ["No", "Yes"])

# Convert user strings to numbers for the model execution
inhibitor_num = 1 if inhibitor == "Yes" else 0
genotype_num = pheno_map[genotype]

# 3. Calculate Live Prediction
if st.button("Calculate Precision Dose"):
    patient_vector = [[age, weight, genotype_num, inhibitor_num, base_dose]]
    prediction = model.predict(patient_vector)[0]
    
    st.markdown("---")
    st.subheader("🤖 AI Clinical Decision Suggestion:")
    
    if prediction == 0:
        st.error("⚠️ REDUCE DOSE: Genetic profile or active drug blocking indicates a dangerously low metabolism. Maintain standard base dose will cause serum toxicity. Reduce starting dose by 50%.")
    elif prediction == 1:
        st.success("✅ MAINTAIN DOSE: Patient metabolic parameters match standard clearance velocity metrics perfectly. Safe to proceed with normal baseline therapeutic dosing protocols.")
    else:
        st.warning("⚡ INCREASE DOSE: Ultra-rapid metabolism detected. The body clears this medication before achieving therapeutic blood concentration levels. Consider a 50% dose escalation or alternate medication.")
