
import streamlit as st
import pandas as pd
import joblib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

st.set_page_config(
    page_title="EstateAI India",
    page_icon="🏡",
    layout="wide"
)

# Store prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load("models/extra_trees_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/final_indian_housing.csv")

model = load_model()
df = load_data()

# Indian number formatting
def indian_format(number):
    number = str(int(round(number)))
    if len(number) <= 3:
        return number

    last = number[-3:]
    remaining = number[:-3]
    parts = []

    while remaining:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]

    return ",".join(parts + [last])


# Create PDF report
def create_pdf(record):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4

    pdf.setTitle("EstateAI Property Prediction Report")

    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(50, height - 60, "EstateAI India")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(
        50, height - 90,
        "House Price Prediction Report"
    )

    pdf.line(50, height - 105, width - 50, height - 105)

    details = [
        ("Date", record["Date"]),
        ("City", record["City"]),
        ("Area", record["Area"]),
        ("BHK", str(record["BHK"])),
        ("Property Type", record["Property Type"]),
        ("Area (Sq. Ft.)", str(record["Square Feet"])),
        ("Posted By", record["Posted By"]),
        ("RERA Approved", record["RERA"]),
        ("Under Construction", record["Construction"]),
        ("Ready To Move", record["Ready"]),
        ("Resale Property", record["Resale"])
    ]

    y = height - 145

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Property Details")
    y -= 30

    pdf.setFont("Helvetica", 11)

    for label, value in details:
        pdf.drawString(60, y, f"{label}: {value}")
        y -= 23

    y -= 10

    pdf.line(50, y, width - 50, y)
    y -= 35

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(50, y, "Estimated Property Price")

    y -= 35

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(
        50, y,
        f"Rs. {record['Price (Lakhs)']:,.2f} Lakh"
    )

    y -= 28

    pdf.setFont("Helvetica", 12)
    pdf.drawString(
        50, y,
        f"Approximately Rs. {indian_format(record['Price (Lakhs)'] * 100000)}"
    )

    y -= 55

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        50, y,
        "This is a machine learning estimate based on historical data."
    )

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# Professional styling
st.markdown("""
<style>
.stApp {
    background: #0b1220;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #142b40, #123e46);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid #245a61;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 42px;
}
.hero p {
    color: #c4dce0;
    font-size: 16px;
}
.section {
    color: #48d6bb;
    font-size: 23px;
    font-weight: 700;
}
.metric {
    background: #172438;
    border: 1px solid #293e54;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}
.metric h2 {
    color: #48d6bb;
}
.metric p {
    color: #bdc9d8;
}
.result {
    background: linear-gradient(120deg, #123e46, #142b40);
    border: 1px solid #28796f;
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}
.result h1 {
    color: #54e3c2;
    font-size: 43px;
}
.result p {
    color: #d5e9e9;
}
div.stButton > button {
    background: #16a085;
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="hero">
    <h1>🏡 EstateAI India</h1>
    <p>Smart Property Valuation Powered by Machine Learning</p>
    <p>Discover estimated residential property prices across India.</p>
</div>
""", unsafe_allow_html=True)


# Dashboard metrics
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric">
        <p>Dataset Properties</p>
        <h2>29,220</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric">
        <p>Model R² Score</p>
        <h2>60.56%</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric">
        <p>Prediction Currency</p>
        <h2>INR ₹</h2>
    </div>
    """, unsafe_allow_html=True)


# Property details
st.write("")
st.markdown(
    '<p class="section">🏠 Property Details</p>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    cities = sorted(df["CITY"].dropna().unique())

    with col1:
        city = st.selectbox("Select City", cities)

        areas = sorted(
            df.loc[df["CITY"] == city, "AREA"].dropna().unique()
        )

        area = st.selectbox("Select Locality", areas)

        bhk = st.selectbox("Number of BHK", range(1, 11))

        property_type = st.selectbox(
            "Property Type", ["BHK", "RK"]
        )

        square_ft = st.number_input(
            "Property Area (Sq. Ft.)",
            min_value=100,
            max_value=10000,
            value=1200,
            step=50
        )

    with col2:
        posted = st.selectbox(
            "Posted By", ["Owner", "Dealer", "Builder"]
        )

        rera = st.selectbox(
            "RERA Approved", ["Yes", "No"]
        )

        construction = st.selectbox(
            "Under Construction", ["No", "Yes"]
        )

        ready = st.selectbox(
            "Ready To Move", ["Yes", "No"]
        )

        resale = st.selectbox(
            "Resale Property", ["Yes", "No"]
        )

    predict = st.button("🔍 Predict Property Price")


# Prediction
if predict:

    input_data = pd.DataFrame([{
        "POSTED_BY": posted,
        "UNDER_CONSTRUCTION": int(construction == "Yes"),
        "RERA": int(rera == "Yes"),
        "BHK_NO.": bhk,
        "BHK_OR_RK": property_type,
        "SQUARE_FT": square_ft,
        "READY_TO_MOVE": int(ready == "Yes"),
        "RESALE": int(resale == "Yes"),
        "AREA": area,
        "CITY": city
    }])

    prediction = float(model.predict(input_data)[0])

    if prediction <= 0:
        st.error("Invalid prediction. Please check property details.")

    else:
        price_rupees = prediction * 100000

        record = {
            "Date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
            "City": city,
            "Area": area,
            "BHK": bhk,
            "Property Type": property_type,
            "Square Feet": square_ft,
            "Posted By": posted,
            "RERA": rera,
            "Construction": construction,
            "Ready": ready,
            "Resale": resale,
            "Price (Lakhs)": round(prediction, 2)
        }

        st.session_state.prediction_history.append(record)

        st.markdown("""
        <p class="section">📊 Property Valuation Result</p>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result">
            <p>Estimated Property Price</p>
            <h1>₹{prediction:,.2f} Lakh</h1>
            <p>Approximately ₹{indian_format(price_rupees)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "This prediction is based on historical dataset records. "
            "It is an estimate, not a guaranteed current market valuation."
        )

        pdf_file = create_pdf(record)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="EstateAI_Property_Report.pdf",
            mime="application/pdf"
        )


# Prediction history
st.write("")
st.divider()

st.markdown(
    '<p class="section">📜 Prediction History</p>',
    unsafe_allow_html=True
)

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="📥 Download History as CSV",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="EstateAI_Prediction_History.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "No prediction history yet. Make your first prediction!"
    )


# Footer
st.write("")
st.divider()

st.markdown("""
<div style="text-align:center; color:#91a0b5; padding:15px;">
    <b>EstateAI India</b><br>
    AI/ML House Price Prediction Project<br>
    Built with Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
