import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Student Retention AI", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {color: #2c3e50;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Retention AI System")
st.caption("⚡ Early Warning System for Student Dropout Prevention")

# -----------------------------
# DATA INPUT
# -----------------------------
st.sidebar.header("📂 Upload Data")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
else:
    st.warning("⚠️ Please upload a student dataset to begin analysis.")
    st.info("Upload a CSV file with columns: Student_ID, Name, Marks, Attendance, Engagement")
    st.stop()
# -----------------------------
# RISK LOGIC
# -----------------------------
def calculate_risk(row):
    score = 0
    if row["Marks"] < 50: score += 40
    if row["Attendance"] < 75: score += 35
    if row["Engagement"] < 5: score += 25

    if score >= 60: return score, "High"
    elif score >= 30: return score, "Medium"
    else: return score, "Low"

data[["Risk Score","Risk Level"]] = data.apply(
    lambda row: pd.Series(calculate_risk(row)), axis=1
)

data["Dropout Risk %"] = data["Risk Score"]

# -----------------------------
# ML MODEL
# -----------------------------
le = LabelEncoder()
data["Risk_Label"] = le.fit_transform(data["Risk Level"])

X = data[["Marks","Attendance","Engagement"]]
y = data["Risk_Label"]

model = RandomForestClassifier()
model.fit(X,y)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard","📋 Students","🔍 Analysis","🎯 Simulator"]
)

# -----------------------------
# DASHBOARD
# -----------------------------
with tab1:
    st.subheader("📊 Institution Overview")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Students", len(data))
    c2.metric("High Risk", len(data[data["Risk Level"]=="High"]))
    c3.metric("Medium Risk", len(data[data["Risk Level"]=="Medium"]))
    c4.metric("Low Risk", len(data[data["Risk Level"]=="Low"]))

    col1,col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.pie(data,names="Risk Level"),use_container_width=True)

    with col2:
        st.plotly_chart(px.scatter(data,x="Attendance",y="Marks",color="Risk Level"),
                        use_container_width=True)

    # Feature importance
    st.subheader("🧠 Feature Importance")
    imp = model.feature_importances_
    st.plotly_chart(px.bar(x=["Marks","Attendance","Engagement"],y=imp),
                    use_container_width=True)

    # Risk ranking (NEW)
    st.subheader("🏆 Top Risk Ranking")
    ranked = data.sort_values(by="Risk Score", ascending=False)
    st.dataframe(ranked[["Name","Risk Score","Risk Level"]].head(10))

# -----------------------------
# STUDENTS TABLE
# -----------------------------
with tab2:
    st.subheader("📋 Student Records")

    filter_risk = st.selectbox("Filter",["All","High","Medium","Low"])
    filtered = data if filter_risk=="All" else data[data["Risk Level"]==filter_risk]

    st.dataframe(filtered[[
        "Student_ID","Name","Marks","Attendance",
        "Engagement","Dropout Risk %","Risk Level"
    ]])

# -----------------------------
# ANALYSIS
# -----------------------------
with tab3:
    st.subheader("🔍 Student Analysis")

    search = st.text_input("Search by Name or ID")

    filtered = data[
        data["Name"].str.contains(search,case=False) |
        data["Student_ID"].str.contains(search,case=False)
    ] if search else data

    if len(filtered)==0:
        st.warning("No student found")
        st.stop()

    st.dataframe(filtered[["Student_ID","Name","Marks","Risk Level"]])

    selected_name = st.selectbox("Select Student",filtered["Name"])
    selected = data[data["Name"]==selected_name].iloc[0]

    m,a,e = selected["Marks"], selected["Attendance"], selected["Engagement"]

    st.metric("Marks",m)
    st.metric("Attendance",f"{a}%")
    st.metric("Engagement",e)

    # Hybrid ML
    if m>=80 and a>=85 and e>=7:
        risk="Low"
    elif m<40 or a<60 or e<4:
        risk="High"
    else:
        pred=model.predict([[m,a,e]])[0]
        risk=le.inverse_transform([pred])[0]

    if risk=="High": st.error("🔴 High Risk")
    elif risk=="Medium": st.warning("🟡 Medium Risk")
    else: st.success("🟢 Low Risk")

    # Main risk factor (NEW)
    st.subheader("🔎 Main Risk Factor")
    if m<50: st.write("Academic Performance")
    elif a<75: st.write("Attendance")
    else: st.write("Engagement")

    # Recommendations
    st.subheader("📌 Recommendations")
    rec=[]
    if m<50: rec.append("Remedial classes")
    if a<75: rec.append("Attendance counseling")
    if e<5: rec.append("Mentor support")
    if not rec: rec=["No action needed"]

    for r in rec:
        st.write("-",r)

    # Parent message
    if st.button("📩 Send Parent Alert"):
        time.sleep(1)
        if risk=="High":
            msg=f"Your child {selected['Name']} is HIGH risk. Immediate action required."
        elif risk=="Medium":
            msg=f"{selected['Name']} shows moderate risk. Please monitor."
        else:
            msg=f"{selected['Name']} is performing well. Keep supporting!"
        st.success("Sent!")
        st.text_area("Message",msg)

# -----------------------------
# SIMULATOR
# -----------------------------
with tab4:
    st.subheader("🎯 Risk Simulator")

    m = st.slider("Marks",0,100,50)
    a = st.slider("Attendance",0,100,75)
    e = st.slider("Engagement",1,10,5)

    if m>=80 and a>=85 and e>=7:
        sim="Low"
    elif m<40 or a<60 or e<4:
        sim="High"
    else:
        sim=le.inverse_transform([model.predict([[m,a,e]])[0]])[0]

    st.success(f"Predicted Risk: {sim}")

    # Intervention simulator (NEW)
    st.subheader("📉 Intervention Effect")

    m2 = st.slider("Improved Marks",0,100,70)
    a2 = st.slider("Improved Attendance",0,100,85)
    e2 = st.slider("Improved Engagement",1,10,7)

    if m2>=80 and a2>=85 and e2>=7:
        improved="Low"
    elif m2<40 or a2<60 or e2<4:
        improved="High"
    else:
        improved=le.inverse_transform([model.predict([[m2,a2,e2]])[0]])[0]

    st.success(f"After Intervention: {improved}")

    # Future scope (ONLY HERE)
    st.subheader("🚀 Future Scope")
    st.write("""
    - Real-time database integration  
    - SMS/Email alerts  
    - Advanced ML models  
    - Mobile application  
    """)

# -----------------------------
# DOWNLOAD
# -----------------------------
st.sidebar.download_button(
    "⬇ Download Data",
    data.to_csv(index=False),
    file_name="students.csv"
)